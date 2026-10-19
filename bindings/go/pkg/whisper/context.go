package whisper

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"
	"time"

	// Bindings
	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type context struct {
	paramsMu sync.Mutex
	n        int
	model    *model
	params   whisper.Params
}

// Make sure context adheres to the interface
var _ Context = (*context)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func newContext(model *model, params whisper.Params) (Context, error) {
	context := new(context)
	context.model = model
	context.params = params

	// Return success
	return context, nil
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func (context *context) SetBestOf(n int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetBestOf(n)
}
func (context *context) SetDetectLanguage(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetDetectLanguage(v)
}
func (context *context) SetSuppressBlank(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetSuppressBlank(v)
}
func (context *context) SetTdrz(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTdrz(v)
}
func (context *context) SetAltN(n int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetAltN(n)
}

// Set the language to use for speech recognition.
func (context *context) SetLanguage(lang string) error {
	ctx, release, err := context.borrow()
	if err != nil {
		return err
	}
	defer release()
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	if ctx.Whisper_is_multilingual() == 0 {
		return ErrModelNotMultilingual
	}

	if lang == "auto" {
		context.params.SetLanguage(-1)
	} else if id := ctx.Whisper_lang_id(lang); id < 0 {
		return ErrUnsupportedLanguage
	} else if err := context.params.SetLanguage(id); err != nil {
		return err
	}
	// Return success
	return nil
}

func (context *context) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Get language
func (context *context) Language() string {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	id := context.params.Language()
	if id == -1 {
		return "auto"
	}
	return whisper.Whisper_lang_str(context.params.Language())
}

func (context *context) DetectedLanguage() string {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Whisper_lang_str(ctx.Whisper_full_lang_id())
}

// Set translate flag
func (context *context) SetTranslate(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTranslate(v)
}

// Voice Activity Detection (VAD)
func (context *context) SetVAD(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVAD(v)
}

func (context *context) SetVADModelPath(path string) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADModelPath(path)
}

func (context *context) SetVADThreshold(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADThreshold(t)
}

func (context *context) SetVADMinSpeechMs(ms int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADMinSpeechMs(ms)
}

func (context *context) SetVADMinSilenceMs(ms int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADMinSilenceMs(ms)
}

func (context *context) SetVADMaxSpeechSec(s float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADMaxSpeechSec(s)
}

func (context *context) SetVADSpeechPadMs(ms int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADSpeechPadMs(ms)
}

func (context *context) SetVADSamplesOverlap(sec float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetVADSamplesOverlap(sec)
}

func (context *context) SetSplitOnWord(v bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetSplitOnWord(v)
}

// Set number of threads to use
func (context *context) SetThreads(v uint) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetThreads(int(v))
}

// Set time offset
func (context *context) SetOffset(v time.Duration) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetOffset(int(v.Milliseconds()))
}

// Set duration of audio to process
func (context *context) SetDuration(v time.Duration) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetDuration(int(v.Milliseconds()))
}

// Set timestamp token probability threshold (~0.01)
func (context *context) SetTokenThreshold(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTokenThreshold(t)
}

// Set timestamp token sum probability threshold (~0.01)
func (context *context) SetTokenSumThreshold(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTokenSumThreshold(t)
}

// Set max segment length in characters
func (context *context) SetMaxSegmentLength(n uint) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetMaxSegmentLength(int(n))
}

// Set token timestamps flag
func (context *context) SetTokenTimestamps(b bool) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTokenTimestamps(b)
}

// Set max tokens per segment (0 = no limit)
func (context *context) SetMaxTokensPerSegment(n uint) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetMaxTokensPerSegment(int(n))
}

// Set audio encoder context
func (context *context) SetAudioCtx(n uint) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetAudioCtx(int(n))
}

// Set maximum number of text context tokens to store
func (context *context) SetMaxContext(n int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetMaxContext(n)
}

// Set Beam Size
func (context *context) SetBeamSize(n int) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetBeamSize(n)
}

// Set Entropy threshold
func (context *context) SetEntropyThold(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetEntropyThold(t)
}

// Set Temperature
func (context *context) SetTemperature(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTemperature(t)
}

// Set the fallback temperature incrementation
// Pass -1.0 to disable this feature
func (context *context) SetTemperatureFallback(t float32) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetTemperatureFallback(t)
}

// Set initial prompt
func (context *context) SetInitialPrompt(prompt string) {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	context.params.SetInitialPrompt(prompt)
}

// ResetTimings resets the mode timings. Should be called before processing
func (context *context) ResetTimings() {
	ctx, release := context.mustBorrow()
	defer release()
	ctx.Whisper_reset_timings()
}

// PrintTimings prints the model timings to stdout.
func (context *context) PrintTimings() {
	ctx, release := context.mustBorrow()
	defer release()
	ctx.Whisper_print_timings()
}

// SystemInfo returns the system information
func (context *context) SystemInfo() string {
	context.paramsMu.Lock()
	defer context.paramsMu.Unlock()
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages.
func (context *context) WhisperLangAutoDetect(offset_ms int, n_threads int) ([]float32, error) {
	ctx, release, err := context.borrow()
	if err != nil {
		return nil, err
	}
	defer release()
	langProbs, err := ctx.Whisper_lang_auto_detect(offset_ms, n_threads)
	if err != nil {
		return nil, err
	}
	return langProbs, nil
}

// Process new sample data and return any errors.
func (context *context) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	return context.process(data, 1, callEncoderBegin, callNewSegment, callProgress)
}

func (context *context) process(data []float32, processors int, callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback, callProgress ProgressCallback) error {
	ctx, release, err := context.borrow()
	if err != nil {
		return err
	}
	defer release()
	// Callback-driven segment policy belongs to this call, not later decodes.
	context.paramsMu.Lock()
	params := context.params
	context.paramsMu.Unlock()
	if callNewSegment != nil {
		params.SetSingleSegment(true)
	}
	var segmentCallback func(int)
	if callNewSegment != nil {
		segmentCallback = func(new int) {
			count := ctx.Whisper_full_n_segments()
			for i := count - new; i < count; i++ {
				callNewSegment(toSegment(ctx, i))
			}
		}
	}
	if processors > 1 {
		err = ctx.Whisper_full_parallel_with_progress(params, data, processors,
			callEncoderBegin, segmentCallback, callProgress)
	} else {
		err = ctx.Whisper_full(params, data, callEncoderBegin, segmentCallback, callProgress)
	}
	if err != nil {
		return err
	}
	// Make all newly decoded segments available to NextSegment.
	context.n = 0
	return nil
}

// Return the next segment of tokens
func (context *context) NextSegment() (Segment, error) {
	ctx, release, err := context.borrow()
	if err != nil {
		return Segment{}, err
	}
	defer release()
	if context.n >= ctx.Whisper_full_n_segments() {
		return Segment{}, io.EOF
	}

	// Populate result
	result := toSegment(ctx, context.n)

	// Increment the cursor
	context.n++

	// Return success
	return result, nil
}

// Test for text tokens
func (context *context) IsText(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	id := whisper.Token(t.Id)
	return id != ctx.Whisper_token_beg() && id != ctx.Whisper_token_sot() &&
		id < ctx.Whisper_token_eot() && id != ctx.Whisper_token_prev() &&
		id != ctx.Whisper_token_solm() && id != ctx.Whisper_token_not()
}

// Test for "begin" token
func (context *context) IsBEG(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_beg()
}

// Test for "start of transcription" token
func (context *context) IsSOT(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_sot()
}

// Test for "end of transcription" token
func (context *context) IsEOT(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_eot()
}

// Test for "start of prev" token
func (context *context) IsPREV(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_prev()
}

// Test for "start of lm" token
func (context *context) IsSOLM(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_solm()
}

// Test for "No timestamps" token
func (context *context) IsNOT(t Token) bool {
	ctx, release := context.mustBorrow()
	defer release()
	return whisper.Token(t.Id) == ctx.Whisper_token_not()
}

// Test for token associated with a specific language
func (context *context) IsLANG(t Token, lang string) bool {
	ctx, release := context.mustBorrow()
	defer release()
	if id := ctx.Whisper_lang_id(lang); id >= 0 {
		return whisper.Token(t.Id) == ctx.Whisper_token_lang(id)
	} else {
		return false
	}
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func toSegment(ctx *whisper.Context, n int) Segment {
	return Segment{
		Num:    n,
		Text:   strings.TrimSpace(ctx.Whisper_full_get_segment_text(n)),
		Start:  time.Duration(ctx.Whisper_full_get_segment_t0(n)) * time.Millisecond * 10,
		End:    time.Duration(ctx.Whisper_full_get_segment_t1(n)) * time.Millisecond * 10,
		Tokens: toTokens(ctx, n),
	}
}

func toTokens(ctx *whisper.Context, n int) []Token {
	result := make([]Token, ctx.Whisper_full_n_tokens(n))
	for i := 0; i < len(result); i++ {
		data := ctx.Whisper_full_get_token_data(n, i)

		result[i] = Token{
			Id:    int(ctx.Whisper_full_get_token_id(n, i)),
			Text:  ctx.Whisper_full_get_token_text(n, i),
			P:     ctx.Whisper_full_get_token_p(n, i),
			Start: time.Duration(data.T0()) * time.Millisecond * 10,
			End:   time.Duration(data.T1()) * time.Millisecond * 10,
		}
	}
	return result
}
