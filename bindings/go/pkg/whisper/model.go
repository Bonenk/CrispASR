package whisper

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	// Bindings
	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type model struct {
	mu   sync.Mutex
	path string
	ctx  *whisper.Context
}

// Make sure model adheres to the interface
var _ Model = (*model)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func New(path string) (Model, error) {
	model := new(model)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	} else if ctx := whisper.Whisper_init(path); ctx == nil {
		return nil, ErrUnableToLoadModel
	} else {
		model.ctx = ctx
		model.path = path
	}

	// Return success
	return model, nil
}

// NewWithDTW loads a Whisper model with experimental DTW token alignment.
// Presets and top-layer counts follow the low-level Whisper_init_with_dtw API.
func NewWithDTW(path string, preset whisper.AlignmentHeadsPreset, nTop int) (Model, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	ctx, err := whisper.Whisper_init_with_dtw(path, preset, nTop)
	if err != nil {
		return nil, err
	}
	return &model{path: path, ctx: ctx}, nil
}

func (model *model) Close() error {
	if model == nil {
		return nil
	}
	if !model.mu.TryLock() {
		return ErrModelBusy
	}
	defer model.mu.Unlock()
	if model.ctx != nil {
		model.ctx.Whisper_free()
		model.ctx = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (model *model) String() string {
	if model == nil {
		return "<whisper.model nil>"
	}
	if !model.mu.TryLock() {
		return "<whisper.model busy>"
	}
	defer model.mu.Unlock()
	str := "<whisper.model"
	if model.ctx != nil {
		str += fmt.Sprintf(" model=%q", model.path)
	}
	return str + ">"
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Return true if model is multilingual (language and translation options are supported)
func (model *model) IsMultilingual() bool {
	ctx, release := model.mustBorrow()
	defer release()
	return ctx.Whisper_is_multilingual() != 0
}

// Return all recognized languages. Initially it is set to auto-detect
func (model *model) Languages() []string {
	ctx, release := model.mustBorrow()
	defer release()
	result := make([]string, 0, whisper.Whisper_lang_max_id()+1)
	for i := 0; i <= whisper.Whisper_lang_max_id(); i++ {
		str := whisper.Whisper_lang_str(i)
		if ctx.Whisper_lang_id(str) >= 0 {
			result = append(result, str)
		}
	}
	return result
}

func (model *model) NewContext() (Context, error) {
	ctx, release, err := model.borrow()
	if err != nil {
		return nil, err
	}
	defer release()

	// Create new context
	params := ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)
	params.SetTranslate(false)
	params.SetPrintSpecial(false)
	params.SetPrintProgress(false)
	params.SetPrintRealtime(false)
	params.SetPrintTimestamps(false)
	params.SetThreads(runtime.NumCPU())
	params.SetNoContext(true)

	// Return new context
	return newContext(model, params)
}
