package whisper

import "errors"

// ParallelProcessor is an optional Context capability. It keeps the existing
// Context interface compatible with third-party implementations.
type ParallelProcessor interface {
	ProcessParallel([]float32, int, EncoderBeginCallback, SegmentCallback, ProgressCallback) error
}

// ProcessParallel splits mono PCM into the requested number of native chunks.
// One processor uses ordinary sequential decoding. Chunk boundaries can change
// the transcript; progress frequency depends on the native decoder. The built-in
// high-level model rejects concurrent/reentrant native operations and callback
// close with ErrModelBusy. Third-party implementations define their own policy.
// Contexts without this capability return an error.
func ProcessParallel(ctx Context, data []float32, processors int, encoder EncoderBeginCallback,
	segment SegmentCallback, progress ProgressCallback) error {
	if processors < 1 || uint64(processors) > 2147483647 {
		return errors.New("processors must be a positive int32")
	}
	provider, ok := ctx.(ParallelProcessor)
	if !ok {
		return errors.New("context does not support parallel processing")
	}
	return provider.ProcessParallel(data, processors, encoder, segment, progress)
}

func (ctx *context) ProcessParallel(data []float32, processors int, encoder EncoderBeginCallback,
	segment SegmentCallback, progress ProgressCallback) error {
	if processors < 1 || uint64(processors) > 2147483647 {
		return errors.New("processors must be a positive int32")
	}
	return ctx.process(data, processors, encoder, segment, progress)
}
