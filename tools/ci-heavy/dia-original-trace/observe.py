"""Read-only original projections/caches and an explicit single-sample fork.

No neural arithmetic or original sampling code is implemented here.
"""
import hashlib


def extend_attention_installer(original, save):
    def install(layers, indices, step, active, original_save):
        handles = original(layers, indices, step, active, original_save)
        try:
            return handles+projection_hooks(layers, indices, active, save)
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
    return install


def projection_hooks(layers, selected, active, save):
    handles = []
    try:
        for index in selected:
            attn = layers[index].self_attention
            for part, heads in [('q', attn.num_query_heads), ('k', attn.num_kv_heads), ('v', attn.num_kv_heads)]:
                name = f'decoder_sa{index}_pre_{part}'
                shape = (2, 1, heads, attn.head_dim)
                def observe(_module, _args, output, name=name, shape=shape):
                    if active():
                        if tuple(output.shape) != shape:
                            raise ValueError('Unexpected actual original projection shape: '+name)
                        save(name, output)
                handles.append(getattr(attn, part+'_proj').register_forward_hook(observe))
        return handles
    except BaseException:
        for handle in handles:
            handle.remove()
        raise


def cross_cache_observer(original, valid, save, layers=18, heads=16, width=128):
    if type(valid) is not int or valid <= 0:
        raise ValueError('Cross prefix length must be positive')
    calls = 0
    def wrapped(*args, **kwargs):
        nonlocal calls
        caches = original(*args, **kwargs)
        calls += 1
        if calls != 1 or len(caches) != layers:
            raise ValueError('Unexpected original cross-cache producer count')
        for index, cache in enumerate(caches):
            for part in ['k', 'v']:
                value = getattr(cache, part)
                if value.ndim != 4 or tuple(value.shape[:2]) != (2, heads) or value.shape[2] < valid or value.shape[3] != width:
                    raise ValueError('Unexpected original cross-cache shape')
                save(f'decoder_cross{index}_{part}', value.transpose(1, 2)[:, :valid])
        return caches
    return wrapped


def single_sample_fork(original, step=978, channel=2, expected=877, replacement=904):
    import torch
    if any(type(v) is not int or v < 0 for v in [step, channel, expected, replacement]) or channel >= 9 or expected == replacement:
        raise ValueError('Invalid single-decision fork recipe')
    state = dict(calls=0, events=[])
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        call = state['calls']; state['calls'] += 1
        if call != step:
            return result
        if tuple(result.shape) != (9,) or result.dtype != torch.int64 or result.device.type != 'cpu':
            raise ValueError('Unexpected actual original sample shape/type/device')
        if int(result[channel]) != expected or state['events']:
            raise ValueError('Canonical original fork sample not reproduced')
        rng_before = torch.get_rng_state().clone()
        used = result.clone()
        used[channel] = replacement
        rng_after = torch.get_rng_state()
        if not torch.equal(rng_before, rng_after):
            raise ValueError('Fork consumed additional RNG')
        state['events'].append(dict(step=call, channel=channel, original=result.tolist(), used=used.tolist(),
            rng_after_original_sha256=hashlib.sha256(rng_before.numpy().tobytes()).hexdigest(),
            rng_after_fork_sha256=hashlib.sha256(rng_after.numpy().tobytes()).hexdigest()))
        return used
    return wrapped, state


def sample_witness(original, step=978):
    import torch
    state = dict(calls=0, events=[])
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        call = state['calls']; state['calls'] += 1
        if call == step:
            if tuple(result.shape) != (9,) or result.dtype != torch.int64 or result.device.type != 'cpu':
                raise ValueError('Unexpected original sample witness shape/type/device')
            state['events'].append(dict(step=call, sampled=result.tolist(),
                rng_after_sample_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        return result
    return wrapped, state
