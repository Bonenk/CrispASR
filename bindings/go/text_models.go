package whisper

/*
#include "crispasr_session.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"strings"
	"sync"
	"unsafe"
)

type textProcessor struct {
	mu     sync.Mutex
	handle unsafe.Pointer
	pcs    bool
}

// TruecaseModel owns a standalone truecaser. Close it when finished.
type TruecaseModel struct{ *textProcessor }

// PCSModel owns standalone punctuation/capitalization/segmentation.
type PCSModel struct{ *textProcessor }

func newTextProcessor(model string, pcs bool) (*textProcessor, error) {
	if strings.IndexByte(model, 0) >= 0 {
		return nil, errors.New("model must not contain NUL")
	}
	path := C.CString(model)
	defer C.free(unsafe.Pointer(path))
	var handle unsafe.Pointer
	if pcs {
		handle = C.crispasr_pcs_init(path)
	} else {
		handle = C.crispasr_truecase_init(path)
	}
	if handle == nil {
		return nil, errors.New("text processor could not open")
	}
	return &textProcessor{handle: handle, pcs: pcs}, nil
}
func TruecaseInit(model string) (*TruecaseModel, error) {
	processor, err := newTextProcessor(model, false)
	if err != nil {
		return nil, err
	}
	return &TruecaseModel{processor}, nil
}
func PCSInit(model string) (*PCSModel, error) {
	processor, err := newTextProcessor(model, true)
	if err != nil {
		return nil, err
	}
	return &PCSModel{processor}, nil
}

// Process copies and frees allocated UTF-8, including successful empty text.
func (p *textProcessor) Process(text string) (string, error) {
	if p == nil {
		return "", errors.New("text processor is closed")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.handle == nil {
		return "", errors.New("text processor is closed")
	}
	if strings.IndexByte(text, 0) >= 0 {
		return "", errors.New("text must not contain NUL")
	}
	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	var output *C.char
	if p.pcs {
		output = (*C.char)(unsafe.Pointer(C.crispasr_pcs_process(p.handle, input)))
	} else {
		output = (*C.char)(unsafe.Pointer(C.crispasr_truecase_process(p.handle, input)))
	}
	if output == nil {
		return "", errors.New("text processing failed")
	}
	if p.pcs {
		defer C.crispasr_pcs_free_text(output)
	} else {
		defer C.crispasr_truecase_free_text(output)
	}
	return C.GoString(output), nil
}

// Close is idempotent and serializes with Process.
func (p *textProcessor) Close() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.handle != nil {
		if p.pcs {
			C.crispasr_pcs_free(p.handle)
		} else {
			C.crispasr_truecase_free(p.handle)
		}
		p.handle = nil
	}
}
