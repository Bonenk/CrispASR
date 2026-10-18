package whisper

/*
#include "crispasr_session.h"
#include <stdlib.h>
struct crispasr_open_params_v1 {
    int abi_version, n_threads, use_gpu, verbosity, flash_attn, n_gpu_layers;
    int reserved[6];
};
*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"strings"
	"unsafe"
)

// SessionOptions controls v2 native loading. Use DefaultSessionOptions for the
// library's GPU/flash defaults; a zero value explicitly disables both.
type SessionOptions struct {
	NThreads       int
	UseGPU         bool
	Verbosity      int
	FlashAttention bool
	GPULayers      int
}

func DefaultSessionOptions() SessionOptions {
	return SessionOptions{NThreads: 4, UseGPU: true, FlashAttention: true, GPULayers: -1}
}
func SessionOpenWithOptions(modelPath, backend string, options SessionOptions) (*CrispasrSession, error) {
	if strings.IndexByte(modelPath, 0) >= 0 || strings.IndexByte(backend, 0) >= 0 {
		return nil, errors.New("model and backend must not contain NUL")
	}
	if options.NThreads < 0 || options.NThreads > math.MaxInt32 || options.Verbosity < 0 || options.Verbosity > math.MaxInt32 || options.GPULayers < -1 || options.GPULayers > math.MaxInt32 {
		return nil, errors.New("invalid session load options")
	}
	threads := options.NThreads
	if threads == 0 {
		threads = 4
	}
	params := C.struct_crispasr_open_params_v1{abi_version: 2, n_threads: C.int(threads), verbosity: C.int(options.Verbosity), n_gpu_layers: C.int(options.GPULayers)}
	if options.UseGPU {
		params.use_gpu = 1
	}
	if options.FlashAttention {
		params.flash_attn = 1
	}
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))
	var be *C.char
	if backend != "" {
		be = C.CString(backend)
		defer C.free(unsafe.Pointer(be))
	}
	handle := C.crispasr_session_open_with_params(path, be, &params)
	if handle == nil {
		return nil, errors.New("session_open_with_params failed")
	}
	return &CrispasrSession{handle: handle}, nil
}

// BackendCapabilities distinguishes an unknown backend from a known empty verb list.
func BackendCapabilities(backend string) ([]string, error) {
	if strings.IndexByte(backend, 0) >= 0 {
		return nil, errors.New("backend must not contain NUL")
	}
	name := C.CString(backend)
	defer C.free(unsafe.Pointer(name))
	for size := 256; size <= 1<<20; size *= 2 {
		buffer := make([]byte, size)
		n := int(C.crispasr_backend_caps_abi(name, (*C.char)(unsafe.Pointer(&buffer[0])), C.int(size)))
		if n == -2 {
			continue
		}
		if n < 0 || n >= size {
			return nil, fmt.Errorf("backend_caps failed (rc=%d)", n)
		}
		if n == 0 {
			return []string{}, nil
		}
		return strings.Split(string(buffer[:n]), ","), nil
	}
	return nil, errors.New("backend capability list exceeds limit")
}
func ListBackendCapabilities() (map[string][]string, error) {
	need := int(C.crispasr_backend_caps_list_abi(nil, 0))
	if need >= 0 || need < -(1<<20) {
		return nil, errors.New("invalid capability size probe")
	}
	for size := -need; size <= 1<<20; {
		buffer := make([]byte, size)
		n := int(C.crispasr_backend_caps_list_abi((*C.char)(unsafe.Pointer(&buffer[0])), C.int(size)))
		if n < 0 {
			if -n <= size {
				return nil, fmt.Errorf("invalid capability size (rc=%d)", n)
			}
			size = -n
			continue
		}
		if n >= size {
			return nil, errors.New("invalid capability byte count")
		}
		out := make(map[string][]string)
		for _, line := range strings.Split(string(buffer[:n]), "\n") {
			if line == "" {
				continue
			}
			fields := strings.SplitN(line, "\t", 2)
			if len(fields) != 2 {
				return nil, errors.New("invalid capability row")
			}
			verbs := []string{}
			if fields[1] != "" {
				verbs = strings.Split(fields[1], ",")
			}
			out[fields[0]] = verbs
		}
		return out, nil
	}
	return nil, errors.New("backend capability list exceeds limit")
}
