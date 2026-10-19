"""Pinned original OpenVoice V2 modules, with owned boundary snapshots.

Set CRISPASR_OV2_UPSTREAM to the pinned upstream checkout and
CRISPASR_OV2_REFERENCE_WAV to a licensed reference. model_dir is the official
checkpoint root (converter/, base_speakers/ses/). No se_extractor, Unidecode,
whisper-timestamped or watermark imports. Native-rate PCM isolates model math
from librosa/native resampler differences, explicitly recorded in metadata.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import numpy as np

DEFAULT_STAGES = [
    'src_pcm', 'ref_pcm', 'src_spec', 'ref_spec', 'ref_enc_layernorm',
    *[f'ref_enc_conv{i}' for i in range(6)], 'ref_enc_gru_out', 'target_se',
    'src_se', 'enc_q_pre', 'enc_q_wn_out', 'enc_q_stats', 'enc_q_z',
    'z_after_flow_fwd', 'z_after_flow_rev', 'dec_conv_pre',
    *[f'dec_stage{i}' for i in range(4)], 'dec_output', 'output_pcm',
]


def dump(model_dir, audio, stages, **kwargs):
    import torch
    import librosa
    root = Path(model_dir)
    upstream = Path(os.environ['CRISPASR_OV2_UPSTREAM']).resolve()
    sys.path.insert(0, str(upstream))
    from openvoice.models import SynthesizerTrn
    from openvoice.mel_processing import spectrogram_torch
    config = json.loads((root / 'converter/config.json').read_text())
    data, hp = config['data'], config['model']
    device = os.environ.get('CRISPASR_OV2_DEVICE', 'cpu')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA reference requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = SynthesizerTrn(len(config.get('symbols', [])), data['filter_length'] // 2 + 1,
                          n_speakers=data['n_speakers'], **hp).to(device).eval()
    checkpoint = torch.load(root / 'converter/checkpoint.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    sr = data['sampling_rate']
    source = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=16000, target_sr=sr)
    reference, _ = librosa.load(os.environ['CRISPASR_OV2_REFERENCE_WAV'], sr=sr, mono=True)
    captures = {}
    handles = []

    def save(name, value, tc=False):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if tc:
            value = value.squeeze(0).T
        captures[name] = np.ascontiguousarray(value).copy()

    def hook(module, name, transform=lambda value: value, tc=False):
        handles.append(module.register_forward_hook(lambda m, a, result: save(name, transform(result), tc)))

    save('raw_audio', source)
    save('reference_audio', reference)
    save('src_pcm', source)
    save('ref_pcm', reference)
    hook(model.ref_enc.layernorm, 'ref_enc_layernorm')
    for i, conv in enumerate(model.ref_enc.convs):
        hook(conv, f'ref_enc_conv{i}', torch.relu)
    hook(model.ref_enc.gru, 'ref_enc_gru_out', lambda out: out[1])
    hook(model.ref_enc.proj, 'target_se')
    hook(model.enc_q.pre, 'enc_q_pre', tc=True)
    hook(model.enc_q.enc, 'enc_q_wn_out', tc=True)
    hook(model.enc_q.proj, 'enc_q_stats', tc=True)
    hook(model.enc_q, 'enc_q_z', lambda out: out[0], tc=True)
    hook(model.dec.conv_pre, 'dec_conv_pre', tc=True)
    n_kernels = len(hp['resblock_kernel_sizes'])
    resblocks = {}
    for i, block in enumerate(model.dec.resblocks):
        def capture_block(m, args, result, i=i):
            resblocks[i] = result.detach().clone()
            if i % n_kernels == n_kernels - 1:
                first = i - n_kernels + 1
                save(f'dec_stage{i // n_kernels}',
                     sum(resblocks[j] for j in range(first, i + 1)) / n_kernels, True)
        handles.append(block.register_forward_hook(capture_block))
    hook(model.dec, 'dec_output')
    handles.append(model.flow.register_forward_hook(
        lambda m, args, kw, result: save('z_after_flow_rev' if kw.get('reverse', False)
                                        else 'z_after_flow_fwd', result, True), with_kwargs=True))

    def spec(pcm):
        y = torch.from_numpy(np.ascontiguousarray(pcm)).unsqueeze(0).to(device)
        return spectrogram_torch(y, data['filter_length'], sr, data['hop_length'],
                                 data['win_length'], center=False)

    speaker = os.environ.get('CRISPASR_OV2_SOURCE_SPEAKER', 'en-default')
    speaker_path = root / 'base_speakers/ses' / (speaker + '.pth')
    tau = float(os.environ.get('CRISPASR_OV2_TAU', '0'))
    if tau not in (0, .3):
        raise ValueError('Reference recipe supports diagnostic tau=0 or upstream tau=0.3')
    # Capture the actual standard-normal input without changing the sampler.
    # A separate C++ RNG cannot be compared by setting the same integer seed.
    original_randn_like = torch.randn_like
    posterior_noise = []
    def capture_noise(*args, **kwargs):
        noise = original_randn_like(*args, **kwargs)
        posterior_noise.append(noise.detach().clone())
        return noise
    try:
        with torch.inference_mode():
            ref_spec = spec(reference)
            save('ref_spec', ref_spec, True)
            target = model.ref_enc(ref_spec.transpose(1, 2)).unsqueeze(-1)
            src_spec = spec(source)
            save('src_spec', src_spec, True)
            src_se = torch.load(speaker_path, map_location=device, weights_only=True).reshape(1, -1, 1)
            save('src_se', src_se)
            lengths = torch.tensor([src_spec.shape[-1]], dtype=torch.long, device=device)
            torch.manual_seed(42)
            torch.randn_like = capture_noise
            try:
                converted = model.voice_conversion(src_spec, lengths, src_se, target, tau=tau)[0][0, 0]
            finally:
                torch.randn_like = original_randn_like
            if len(posterior_noise) != 1:
                raise RuntimeError('Unexpected posterior sampling count')
            if tau:
                save('enc_q_noise', posterior_noise[0], True)
            # Remove only our diagnostic hooks, then repeat the untouched
            # inference with the same random seed and actual model inputs.
            for handle in handles:
                handle.remove()
            torch.manual_seed(42)
            pristine = model.voice_conversion(src_spec, lengths, src_se, target, tau=tau)[0][0, 0]
            if not torch.equal(converted, pristine):
                raise RuntimeError('Capture instrumentation changed original output')
            output = converted.detach().float().cpu().numpy().copy()
            peak = np.max(np.abs(output))
            if peak > .01:
                output *= np.float32(.95) / peak
            save('output_pcm', output)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(DEFAULT_STAGES) - captures.keys()
    if missing:
        raise RuntimeError('Original blueprint did not execute stages: ' + str(sorted(missing)))
    # Keep metadata portable. Inputs and boundaries are self-contained in GGUF.
    captures.update(ov2_source_speaker=speaker, ov2_sample_rate=str(sr), ov2_tau=str(tau),
                    ov2_capture_neutral='byte-identical pristine replay',
                    ov2_recipe=f'native-rate raw reference; no VAD; watermark disabled; tau={tau}; shared posterior noise',
                    ov2_checkpoint_sha256=hashlib.file_digest(
                        (root / 'converter/checkpoint.pth').open('rb'), 'sha256').hexdigest())
    return captures
