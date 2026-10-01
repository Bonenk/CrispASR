"""Independent Index-Echo reference using its released inference class.

Pin the source package; do not run a GGUF reconstruction as the oracle.
Capture full audio stages, last prompt-token decoder states, logits and text.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_STAGES = ['mel_spectrogram', 'encoder_input', 'encoder_output', 'connector_output',
                  'prompt_ids', 'llm_logits', 'generated_ids']


def dump(model_dir, audio, stages, **kwargs):
    import soundfile as sf
    import torch
    torch.set_num_threads(int(os.getenv('INDEX_ECHO_REF_THREADS', '4')))
    torch.set_grad_enabled(False)
    root = Path(model_dir)
    spec = importlib.util.spec_from_file_location('index_echo_blueprint', root / 'infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.AudioTransModel(str(root), device='cpu', dtype=torch.float32)
    values = {}
    handles = []
    def hook(name, last=False, transform=None):
        def capture(mod, inputs, output):
            v = output[0] if isinstance(output, tuple) else output
            if last:
                v = v[:, -1, :]
            if transform:
                v = transform(v)
            values[name] = v.detach().float().cpu().numpy().copy()
        return capture
    for i, layer in enumerate(model.tower.layers):
        handles.append(layer.register_forward_hook(hook(f'encoder_layer_{i}')))
    def first_input(mod, inputs):
        values['encoder_input'] = inputs[0].detach().float().cpu().numpy().copy()
    handles.append(model.tower.layers[0].register_forward_pre_hook(first_input))
    handles.append(model.tower.ln_post.register_forward_hook(hook('ln_post_out')))
    handles.append(model.tower.proj1.register_forward_hook(hook('proj1_out')))
    for i in range(1, 4):
        handles.append(getattr(model.tower, f'conv2d{i}').register_forward_hook(
            hook(f'conv{i}_out', transform=torch.nn.functional.gelu)))
    handles.append(model.tower.proj2.register_forward_hook(hook('encoder_output')))
    with tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR')) as tmp:
        wav = Path(tmp) / 'input.wav'
        sf.write(wav, audio, 16000, subtype='FLOAT')
        # The exact released frontend with its attention-mask frame count.
        f = model.fe(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        frames = int(f.attention_mask.sum(-1)[0])
        values['mel_spectrogram'] = f.input_features[0][:, :frames].float().numpy().copy()
        emb = model.encode_audio(str(wav))
    values['connector_output'] = emb.float().numpy().copy()
    for h in handles:
        h.remove()
    handles = []
    lang = os.getenv('INDEX_ECHO_TARGET_LANG', 'en')
    content = module.AUDIO_START + module.AUDIO_PAD * emb.shape[0] + module.AUDIO_END + '\n' + module.INSTR[lang]
    prompt = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    ids = model.tok(prompt, return_tensors='pt').input_ids
    x = model.llm.get_input_embeddings()(ids).clone()
    mask = ids == model.pad_id
    assert int(mask.sum()) == emb.shape[0]
    x[mask] = emb.to(x.dtype)
    values['prompt_ids'] = ids.numpy().astype(np.int32)
    for i, layer in enumerate(model.llm.model.layers):
        handles.append(layer.register_forward_hook(hook(f'llm_block_{i}', last=True)))
    out = model.llm(inputs_embeds=x, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    values['llm_logits'] = out.logits[0, -1].float().numpy().copy()
    for h in handles:
        h.remove()
    trace_cache = out.past_key_values
    del out
    generated = model.llm.generate(inputs_embeds=x, attention_mask=torch.ones_like(ids),
        max_new_tokens=int(os.getenv('INDEX_ECHO_REF_MAX_TOKENS', '2000')), do_sample=False,
        eos_token_id=[model.tok.eos_token_id, model.im_end], pad_token_id=model.tok.eos_token_id)
    values['generated_ids'] = generated.numpy().astype(np.int32)
    values['generated_text'] = model.tok.decode(generated[0], skip_special_tokens=True).strip()
    # Replay the reference's own greedy IDs through the saved initial cache.
    # This separates recurrent/KV errors from divergent sampling decisions.
    trace = [values['llm_logits']]
    for i in range(min(16, generated.shape[-1]) - 1):
        step = model.llm(input_ids=generated[:, i:i + 1], past_key_values=trace_cache,
                         use_cache=True, logits_to_keep=1)
        trace_cache = step.past_key_values
        trace.append(step.logits[0, -1].float().numpy().copy())
    values['teacherforced_logits'] = np.stack(trace)
    return values
