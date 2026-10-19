package whisper

import (
	"errors"
	native "github.com/CrispStrobe/CrispASR/bindings/go"
)

// TokenSnapshotProvider is an optional Context capability, leaving the existing
// Context interface and Token struct source-compatible with older consumers.
type TokenSnapshotProvider interface {
	TokenSnapshots(segment int) ([]native.WhisperTokenSnapshot, error)
}

// TokenSnapshots retrieves owned IDs, raw bytes, native-unit timing/DTW,
// probabilities and alternatives. Contexts without the capability return errors.
func TokenSnapshots(ctx Context, segment int) ([]native.WhisperTokenSnapshot, error) {
	provider, ok := ctx.(TokenSnapshotProvider)
	if !ok {
		return nil, errors.New("context does not support token snapshots")
	}
	return provider.TokenSnapshots(segment)
}

func (ctx *context) TokenSnapshots(segment int) ([]native.WhisperTokenSnapshot, error) {
	if ctx == nil || ctx.model == nil || ctx.model.ctx == nil {
		return nil, ErrInternalAppError
	}
	return ctx.model.ctx.Whisper_token_snapshots(segment)
}
