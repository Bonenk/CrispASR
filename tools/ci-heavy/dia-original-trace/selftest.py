"""Tiny actual pinned Attention/KVCache/sampler contracts; no full model."""
import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('trace_observers', HERE/'observe.py')
observe = importlib.util.module_from_spec(spec); spec.loader.exec_module(observe)
SOURCE = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blueprint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    # Use the pinned test's tiny original neural model (only its DAC is a stub).
    # Exercise the reference's real function-local installer import as the worker does.
    sys.path.insert(0, str(args.reference/'tools'))
    test_path = args.reference/'tests/test_dia_reference_capture.py'
    spec = importlib.util.spec_from_file_location('tiny_original_fixture', test_path)
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    import os
    os.environ['DIA_SOURCE'] = str(args.blueprint)
    fixture.DiaCaptureTest.setUpClass()
    import dia_attention_capture as attention_capture
    import dia_decoder_capture as boundary_capture
    from reference_backends import dia_tts_reference as reference
    dia = fixture.DiaCaptureTest.dia
    extras = {}
    def own(name, value):
        assert name not in extras
        extras[name] = value.detach().cpu().numpy().copy()
    kwargs = dict(max_tokens=32, capture_step=7)
    baseline = reference.capture(dia, '[S1] Tiny capture test.', **kwargs)
    # capture restores bound methods on the instance. The hosted arms each
    # construct a fresh model; reproduce that here for the class-level observer.
    fixture.DiaCaptureTest.setUpClass()
    dia = fixture.DiaCaptureTest.dia
    selected = (0, 1)
    stages = set(baseline) | set(attention_capture.attention_stages(selected)) | set(boundary_capture.boundary_stages(selected))
    original_cross = dia.model.decoder.__class__.precompute_cross_attn_cache
    valid = baseline['encoder_tokens'].shape[1]
    with mock.patch.object(attention_capture, 'install_attention_hooks',
                           observe.extend_attention_installer(attention_capture.install_attention_hooks, own)), \
         mock.patch.object(dia.model.decoder.__class__, 'precompute_cross_attn_cache',
                           observe.cross_cache_observer(original_cross, valid, own, layers=2, heads=2, width=4)):
        captured = reference.capture(dia, '[S1] Tiny capture test.', stages=stages,
                                     attention_layers=selected, boundary_layers=selected, **kwargs)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in dia.model.modules())
    fixture.DiaCaptureTest.setUpClass()
    dia = fixture.DiaCaptureTest.dia
    repeated = reference.capture(dia, '[S1] Tiny capture test.', **kwargs)
    for name, value in baseline.items():
        assert value.tobytes() == captured[name].tobytes() == repeated[name].tobytes(), name
    assert len(extras) == 10, sorted(extras)
    assert set(stages) == set(captured), (sorted(set(stages)-set(captured)), sorted(set(captured)-set(stages)))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in dia.model.modules())
    print('DIA_ORIGINAL_TRACE_WIRING_PASS: actual tiny original generation, simultaneous boundaries/projections/caches, OFF/ON/OFF exact', flush=True)
    assert torch.__version__ == '2.6.0+cpu'; torch.set_num_threads(4)
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Tensor=torch.Tensor, DiaConfig=object)
    for filename, selected in [('state.py', {'KVCache'}),
                              ('layers.py', {'_normalize_axes', 'DenseGeneral', 'RotaryEmbedding', 'Attention'}),
                              ('model.py', {'_sample_next_token'})]:
        file = args.blueprint/'dia'/filename
        assert file.read_bytes() == subprocess.check_output(['git', 'show', SOURCE+':dia/'+filename], cwd=args.blueprint)
        nodes = [n for n in ast.parse(file.read_text()).body if getattr(n, 'name', None) in selected]
        assert len(nodes) == len(selected)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), ns)
    cfg = types.SimpleNamespace(model=types.SimpleNamespace(rope_min_timescale=1, rope_max_timescale=10000))
    attn = ns['Attention'](cfg, 8, 8, 2, 1, 4, torch.float32).eval()
    with torch.no_grad():
        for parameter in attn.parameters():
            parameter.copy_(torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)/parameter.numel())
    x = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)/16
    pos = torch.full((2, 1), 7, dtype=torch.int32)
    saved = {}
    def save(name, value):
        assert name not in saved and value.dtype == torch.float32 and torch.isfinite(value).all()
        saved[name] = value.detach().cpu().numpy().copy()
    with torch.inference_mode():
        expected = {part: getattr(attn, part+'_proj')(x).numpy().copy() for part in ['q', 'k', 'v']}
        off = attn(x, x, pos).clone()
        rng = torch.get_rng_state().clone()
        handles = observe.projection_hooks([types.SimpleNamespace(self_attention=attn)], [0], lambda: True, save)
        on = attn(x, x, pos).clone()
        for handle in handles: handle.remove()
        repeat = attn(x, x, pos).clone()
        assert torch.equal(off, on) and torch.equal(off, repeat) and torch.equal(rng, torch.get_rng_state())
    assert set(saved) == {f'decoder_sa0_pre_{part}' for part in expected}
    for part, value in expected.items():
        assert saved['decoder_sa0_pre_'+part].tobytes() == value.tobytes()
    # Original cache objects are returned unchanged; only a valid prefix is copied.
    caches = [ns['KVCache'].from_kv(torch.arange(80, dtype=torch.float32).reshape(2, 2, 5, 4)+i*100,
                                   -torch.arange(80, dtype=torch.float32).reshape(2, 2, 5, 4)-i*100) for i in range(2)]
    before = [(cache.k.clone(), cache.v.clone(), cache.k.data_ptr(), cache.v.data_ptr()) for cache in caches]
    wrapped = observe.cross_cache_observer(lambda: caches, 3, save, layers=2, heads=2, width=4)
    assert wrapped() is caches
    for index, (cache, (k, v, kp, vp)) in enumerate(zip(caches, before)):
        assert cache.k.data_ptr() == kp and cache.v.data_ptr() == vp
        assert torch.equal(cache.k, k) and torch.equal(cache.v, v)
        for part, full in [('k', k), ('v', v)]:
            assert saved[f'decoder_cross{index}_{part}'].tobytes() == full.transpose(1, 2)[:, :3].numpy().tobytes()
    sampler = ns['_sample_next_token']
    logits = torch.arange(9*64, dtype=torch.float32).reshape(9, 64)/100
    def run_sample(): return sampler(logits, 1.2, .95, 7)
    torch.manual_seed(42); baseline = [run_sample().clone() for _ in range(5)]; end_rng = torch.get_rng_state().clone()
    expected = int(baseline[1][2]); replacement = (expected+1) % 64
    originals = []
    def recorded():
        value = run_sample(); originals.append(value); return value
    witnessed, witness = observe.sample_witness(recorded, step=1)
    fork, state = observe.single_sample_fork(witnessed, step=1, expected=expected, replacement=replacement)
    torch.manual_seed(42); forked = [fork().clone() for _ in range(5)]
    assert torch.equal(end_rng, torch.get_rng_state()) and state['calls'] == 5 and len(state['events']) == 1
    for i in range(5):
        assert torch.equal(originals[i], baseline[i])
        wanted = baseline[i].clone()
        if i == 1: wanted[2] = replacement
        assert torch.equal(forked[i], wanted)
    event = state['events'][0]
    assert event['rng_after_original_sha256'] == event['rng_after_fork_sha256']
    assert witness['events'][0]['sampled'] == baseline[1].tolist()
    assert witness['events'][0]['rng_after_sample_sha256'] == event['rng_after_original_sha256']
    assert [i for i, (a, b) in enumerate(zip(event['original'], event['used'])) if a != b] == [2]
    rejects = 0
    for bad in [dict(step=-1), dict(channel=9), dict(expected=1, replacement=1)]:
        try: observe.single_sample_fork(run_sample, **bad)
        except ValueError: rejects += 1
        else: raise AssertionError('Invalid fork recipe accepted')
    wrong, _ = observe.single_sample_fork(run_sample, step=1, expected=(expected+2) % 64, replacement=replacement)
    torch.manual_seed(42); wrong()
    try: wrong()
    except ValueError: rejects += 1
    else: raise AssertionError('Wrong canonical sample accepted')
    for call in [lambda: wrapped(), lambda: observe.cross_cache_observer(lambda: caches, 6, save, 2, 2, 4)()]:
        try: call()
        except ValueError: rejects += 1
        else: raise AssertionError('Wrong cache producer/prefix accepted')
    print(f'DIA_ORIGINAL_TRACE_CONTRACTS_PASS: actual pinned Attention OFF/ON/OFF and KVCache identity, actual sampler single fork/RNG exact, {rejects} rejected controls')


if __name__ == '__main__':
    main()
