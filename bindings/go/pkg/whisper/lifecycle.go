package whisper

import (
	"errors"
	native "github.com/CrispStrobe/CrispASR/bindings/go"
)

// ErrModelBusy means another operation owns this high-level model. Retrying
// after that call returns is safe. Close and decode never wait on callbacks.
var ErrModelBusy = errors.New("whisper model is busy")

func (model *model) borrow() (*native.Context, func(), error) {
	if model == nil {
		return nil, nil, ErrInternalAppError
	}
	if !model.mu.TryLock() {
		return nil, nil, ErrModelBusy
	}
	if model.ctx == nil {
		model.mu.Unlock()
		return nil, nil, ErrInternalAppError
	}
	return model.ctx, model.mu.Unlock, nil
}

func (context *context) borrow() (*native.Context, func(), error) {
	if context == nil {
		return nil, nil, ErrInternalAppError
	}
	return context.model.borrow()
}

// Existing methods without an error return reject lifecycle misuse by panic.
// Error-returning operations (including Close) return the same sentinel.
func (model *model) mustBorrow() (*native.Context, func()) {
	ctx, release, err := model.borrow()
	if err != nil {
		panic(err)
	}
	return ctx, release
}
func (context *context) mustBorrow() (*native.Context, func()) {
	ctx, release, err := context.borrow()
	if err != nil {
		panic(err)
	}
	return ctx, release
}
