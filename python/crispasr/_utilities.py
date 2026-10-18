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
