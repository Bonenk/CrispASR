"""Shared-library utilities and standalone text processors."""
import ctypes as C

from ._session_extra import function


def _library(lib_path=None):
    from ._binding import _find_lib
    return C.CDLL(lib_path or _find_lib())


def set_gpu_backend(name, *, lib_path=None):
    """Set process-wide GPU preference before opening model sessions."""
    function(_library(lib_path), "crispasr_set_gpu_backend", [C.c_char_p], None)(
        None if name is None else name.encode("utf-8"))


def c_api_version(*, lib_path=None):
    return (function(_library(lib_path), "crispasr_c_api_version", [], C.c_char_p)() or b"").decode()


def dart_helpers_version(*, lib_path=None):
    """Report native helper ABI version (useful for loaded-library diagnostics)."""
    return (function(_library(lib_path), "crispasr_dart_helpers_version", [], C.c_char_p)() or b"").decode()


def backend_capabilities(backend=None, *, lib_path=None):
    """Return declared verbs for one backend, or a mapping for all backends.

    This reports adapter capabilities; available_backends() separately reports
    which backends were compiled into the loaded library.
    """
    lib = _library(lib_path)
    if backend is None:
        fn = function(lib, "crispasr_backend_caps_list_abi", [C.c_char_p, C.c_int32], C.c_int)
        required = fn(None, 0)
        if required >= 0:
            raise RuntimeError("invalid capability buffer size probe")
        buf = C.create_string_buffer(-required)
        rc = fn(buf, len(buf))
        if rc < 0:
            raise RuntimeError(f"backend capabilities failed (rc={rc})")
        return {name: tuple(filter(None, caps.split(',')))
                for name, caps in (line.split('\t', 1) for line in buf.value.decode().splitlines())}
    fn = function(lib, "crispasr_backend_caps_abi", [C.c_char_p, C.c_char_p, C.c_int32], C.c_int)
    size = 256
    while size <= 1048576:
        buf = C.create_string_buffer(size)
        rc = fn(backend.encode(), buf, size)
        if rc == -3:
            raise ValueError(f"unknown backend {backend!r}")
        if rc == -2:
            size *= 2
            continue
        if rc < 0:
            raise RuntimeError(f"backend capabilities failed (rc={rc})")
        return tuple(filter(None, buf.value.decode().split(',')))
    raise RuntimeError("capability response exceeds 1 MiB")


class _TextModel:
    _prefix = None

    def __init__(self, model_path, lib_path=None):
        self._handle = None
        self._lib = _library(lib_path)
        self._init = function(self._lib, self._prefix + '_init', [C.c_char_p], C.c_void_p)
        self._process = function(self._lib, self._prefix + '_process', [C.c_void_p, C.c_char_p], C.c_void_p)
        self._free_text = function(self._lib, self._prefix + '_free_text', [C.c_void_p], None)
        self._free = function(self._lib, self._prefix + '_free', [C.c_void_p], None)
        self._handle = self._init(str(model_path).encode('utf-8'))
        if not self._handle:
            raise RuntimeError(f"failed to load {self._prefix} model")

    def process(self, text):
        if not self._handle:
            raise RuntimeError("text model is closed")
        ptr = self._process(self._handle, text.encode('utf-8'))
        if not ptr:
            raise RuntimeError(f"{self._prefix} processing failed")
        try:
            return C.string_at(ptr).decode('utf-8')
        finally:
            self._free_text(ptr)

    def close(self):
        if getattr(self, '_handle', None):
            self._free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


class TruecaseModel(_TextModel):
    """Standalone truecasing with a loaded native truecase model."""
    _prefix = 'crispasr_truecase'


class PCSModel(_TextModel):
    """Standalone punctuation/capitalization/segmentation model."""
    _prefix = 'crispasr_pcs'


from dataclasses import dataclass
import operator
import os
import numpy as np
from ._session_extra import F, I, mono, pointer, copy_view


@dataclass
class AudioData:
    """Owned mono [frame] or interleaved stereo [frame,2] PCM and its rate.

    source_channels preserves the stereo decoder's 1/2 source-channel flag;
    stereo output always has two columns, including duplicated mono sources.
    """
    samples: np.ndarray
    sample_rate: int
    source_channels: int = 1


def load_audio(path, *, sample_rate=None, lib_path=None):
    """Decode with the native multi-format loader; default output is 16 kHz mono."""
    lib = _library(lib_path)
    free = function(lib, 'crispasr_audio_free', [F], None)
    pcm, n, rate = F(), C.c_int(), C.c_int()
    if sample_rate is None:
        fn = function(lib, 'crispasr_audio_load', [C.c_char_p, C.POINTER(F), I, I], C.c_int)
        args = [os.fsencode(path)]
    else:
        sample_rate = operator.index(sample_rate)
        if sample_rate <= 0 or sample_rate > 2147483647:
            raise ValueError('sample_rate must be a positive C integer')
        fn = function(lib, 'crispasr_audio_load_at_rate', [C.c_char_p, C.c_int, C.POINTER(F), I, I], C.c_int)
        args = [os.fsencode(path), sample_rate]
    try:
        rc = fn(*args, C.byref(pcm), C.byref(n), C.byref(rate))
        if rc != 0:
            raise RuntimeError(f'audio decode failed (rc={rc})')
        return AudioData(copy_view(pcm, (n.value,)), rate.value)
    finally:
        if pcm:
            free(pcm)


def load_audio_stereo(path, *, lib_path=None):
    """Decode two owned PCM columns at 16 kHz, preserving the source channel flag."""
    lib = _library(lib_path)
    free = function(lib, 'crispasr_audio_free', [F], None)
    fn = function(lib, 'crispasr_audio_load_stereo',
                  [C.c_char_p, C.POINTER(F), C.POINTER(F), I, I, I], C.c_int)
    left, right, n, rate, channels = F(), F(), C.c_int(), C.c_int(), C.c_int()
    try:
        rc = fn(os.fsencode(path), C.byref(left), C.byref(right), C.byref(n), C.byref(rate), C.byref(channels))
        if rc != 0:
            raise RuntimeError(f'stereo audio decode failed (rc={rc})')
        samples = np.column_stack((copy_view(left, (n.value,)), copy_view(right, (n.value,))))
        return AudioData(samples, rate.value, channels.value)
    finally:
        if left:
            free(left)
        if right and C.cast(right, C.c_void_p).value != C.cast(left, C.c_void_p).value:
            free(right)


def _container_result(lib, call):
    free = function(lib, 'crispasr_c2pa_free', [C.c_void_p], None)
    n = C.c_size_t()
    ptr = call(C.byref(n))
    try:
        if not ptr:
            raise RuntimeError('container operation failed, unavailable, or unsupported format')
        return C.string_at(ptr, n.value)
    finally:
        if ptr:
            free(ptr)


def pcm_to_wav(pcm, sample_rate=16000, *, lib_path=None):
    """Wrap mono PCM as 16-bit WAV carrying native provenance metadata."""
    pcm = mono(pcm)
    sample_rate = operator.index(sample_rate)
    if not len(pcm) or not 0 < sample_rate <= 2147483647:
        raise ValueError('nonempty mono PCM and a positive C integer sample rate are required')
    lib = _library(lib_path)
    fn = function(lib, 'crispasr_pcm_to_wav', [F, C.c_int, C.c_int, C.POINTER(C.c_size_t)], C.c_void_p)
    return _container_result(lib, lambda n: fn(pointer(pcm), len(pcm), sample_rate, n))


def c2pa_sign(data, format='audio/wav', *, cert_path=None, key_path=None, lib_path=None):
    """Sign container bytes with native C2PA; unavailable/failed signing raises.

    Optional PEM paths must be supplied together; otherwise the native bundled
    self-signed certificate is used. This does not represent a trusted identity.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('data must be container bytes')
    data = bytes(data)
    if not data or not format:
        raise ValueError('nonempty container bytes and MIME format are required')
    if (cert_path is None) != (key_path is None):
        raise ValueError('cert_path and key_path must be supplied together')
    lib = _library(lib_path)
    fn = function(lib, 'crispasr_c2pa_sign',
                  [C.c_void_p, C.c_size_t, C.c_char_p, C.c_char_p, C.c_char_p, C.POINTER(C.c_size_t)], C.c_void_p)
    buffer = C.create_string_buffer(data)
    return _container_result(lib, lambda n: fn(buffer, len(data), format.encode('utf-8'),
                                              None if cert_path is None else os.fsencode(cert_path),
                                              None if key_path is None else os.fsencode(key_path), n))
