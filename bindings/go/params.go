package whisper

import (
	"fmt"
	"strings"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#include <crispasr.h>
#include <stdlib.h>
*/
import "C"

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func (p *Params) SetTranslate(v bool) {
	p.native.translate = toBool(v)
}

func (p *Params) SetSplitOnWord(v bool) {
	p.native.split_on_word = toBool(v)
}

func (p *Params) SetNoContext(v bool) {
	p.native.no_context = toBool(v)
}

func (p *Params) SetSingleSegment(v bool) {
	p.native.single_segment = toBool(v)
}

func (p *Params) SetPrintSpecial(v bool) {
	p.native.print_special = toBool(v)
}

func (p *Params) SetPrintProgress(v bool) {
	p.native.print_progress = toBool(v)
}

func (p *Params) SetPrintRealtime(v bool) {
	p.native.print_realtime = toBool(v)
}

func (p *Params) SetPrintTimestamps(v bool) {
	p.native.print_timestamps = toBool(v)
}

// Voice Activity Detection (VAD)
func (p *Params) SetVAD(v bool) {
	p.native.vad = toBool(v)
}

func (p *Params) SetVADModelPath(path string) {
	p.vadModelPath = path
	p.vadModelPathSet = true
}

func (p *Params) SetVADThreshold(t float32) {
	p.native.vad_params.threshold = C.float(t)
}

func (p *Params) SetVADMinSpeechMs(ms int) {
	p.native.vad_params.min_speech_duration_ms = C.int(ms)
}

func (p *Params) SetVADMinSilenceMs(ms int) {
	p.native.vad_params.min_silence_duration_ms = C.int(ms)
}

func (p *Params) SetVADMaxSpeechSec(s float32) {
	p.native.vad_params.max_speech_duration_s = C.float(s)
}

func (p *Params) SetVADSpeechPadMs(ms int) {
	p.native.vad_params.speech_pad_ms = C.int(ms)
}

func (p *Params) SetVADSamplesOverlap(sec float32) {
	p.native.vad_params.samples_overlap = C.float(sec)
}

// Set language id
func (p *Params) SetLanguage(lang int) error {
	if lang == -1 {
		p.native.language = nil
		return nil
	}
	str := C.whisper_lang_str(C.int(lang))
	if str == nil {
		return ErrInvalidLanguage
	} else {
		p.native.language = str
	}
	return nil
}

// Get language id
func (p *Params) Language() int {
	if p.native.language == nil {
		return -1
	}
	return int(C.whisper_lang_id(p.native.language))
}

// Threads available
func (p *Params) Threads() int {
	return int(p.native.n_threads)
}

// Set number of threads to use
func (p *Params) SetThreads(threads int) {
	p.native.n_threads = C.int(threads)
}

// Set start offset in ms
func (p *Params) SetOffset(offset_ms int) {
	p.native.offset_ms = C.int(offset_ms)
}

// Set audio duration to process in ms
func (p *Params) SetDuration(duration_ms int) {
	p.native.duration_ms = C.int(duration_ms)
}

// Set timestamp token probability threshold (~0.01)
func (p *Params) SetTokenThreshold(t float32) {
	p.native.thold_pt = C.float(t)
}

// Set timestamp token sum probability threshold (~0.01)
func (p *Params) SetTokenSumThreshold(t float32) {
	p.native.thold_ptsum = C.float(t)
}

// Set max segment length in characters
func (p *Params) SetMaxSegmentLength(n int) {
	p.native.max_len = C.int(n)
}

func (p *Params) SetTokenTimestamps(b bool) {
	p.native.token_timestamps = toBool(b)
}

// Set max tokens per segment (0 = no limit)
func (p *Params) SetMaxTokensPerSegment(n int) {
	p.native.max_tokens = C.int(n)
}

// Set audio encoder context
func (p *Params) SetAudioCtx(n int) {
	p.native.audio_ctx = C.int(n)
}

func (p *Params) SetMaxContext(n int) {
	p.native.n_max_text_ctx = C.int(n)
}

func (p *Params) SetBeamSize(n int) {
	p.native.beam_search.beam_size = C.int(n)
}

func (p *Params) SetEntropyThold(t float32) {
	p.native.entropy_thold = C.float(t)
}

func (p *Params) SetTemperature(t float32) {
	p.native.temperature = C.float(t)
}

// Sets the fallback temperature incrementation
// Pass -1.0 to disable this feature
func (p *Params) SetTemperatureFallback(t float32) {
	p.native.temperature_inc = C.float(t)
}

// Set initial prompt
func (p *Params) SetInitialPrompt(prompt string) {
	p.initialPrompt = prompt
	p.initialPromptSet = true
}

func (p *Params) SetCarryInitialPrompt(v bool) {
	p.native.carry_initial_prompt = toBool(v)
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func toBool(v bool) C.bool {
	if v {
		return C.bool(true)
	}
	return C.bool(false)
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (p *Params) String() string {
	str := "<whisper.params"
	str += fmt.Sprintf(" strategy=%v", p.native.strategy)
	str += fmt.Sprintf(" n_threads=%d", p.native.n_threads)
	if p.native.language != nil {
		str += fmt.Sprintf(" language=%s", C.GoString(p.native.language))
	}
	str += fmt.Sprintf(" n_max_text_ctx=%d", p.native.n_max_text_ctx)
	str += fmt.Sprintf(" offset_ms=%d", p.native.offset_ms)
	str += fmt.Sprintf(" duration_ms=%d", p.native.duration_ms)
	str += fmt.Sprintf(" audio_ctx=%d", p.native.audio_ctx)
	prompt := p.initialPrompt
	if !p.initialPromptSet {
		prompt = C.GoString(p.native.initial_prompt)
	}
	str += fmt.Sprintf(" initial_prompt=%s", prompt)
	str += fmt.Sprintf(" entropy_thold=%f", p.native.entropy_thold)
	str += fmt.Sprintf(" temperature=%f", p.native.temperature)
	str += fmt.Sprintf(" temperature_inc=%f", p.native.temperature_inc)
	str += fmt.Sprintf(" beam_size=%d", p.native.beam_search.beam_size)
	if p.native.translate {
		str += " translate"
	}
	if p.native.no_context {
		str += " no_context"
	}
	if p.native.single_segment {
		str += " single_segment"
	}
	if p.native.print_special {
		str += " print_special"
	}
	if p.native.print_progress {
		str += " print_progress"
	}
	if p.native.print_realtime {
		str += " print_realtime"
	}
	if p.native.print_timestamps {
		str += " print_timestamps"
	}
	if p.native.token_timestamps {
		str += " token_timestamps"
	}
	if p.native.carry_initial_prompt {
		str += " carry_initial_prompt"
	}

	return str + ">"
}

// SetBestOf updates the native positive best-of control; nonpositive values
// leave it unchanged, matching crispasr_params_set_best_of.
func (p *Params) SetBestOf(n int) {
	if n > 0 && uint64(n) <= 2147483647 {
		p.native.greedy.best_of = C.int(n)
	}
}
func (p *Params) SetDetectLanguage(v bool) { p.native.detect_language = toBool(v) }
func (p *Params) SetSuppressBlank(v bool)  { p.native.suppress_blank = toBool(v) }
func (p *Params) SetTdrz(v bool)           { p.native.tdrz_enable = toBool(v) }

// SetAltN clamps the candidate count to [0,32], matching the native adapter.
func (p *Params) SetAltN(n int) {
	if n < 0 {
		n = 0
	}
	if n > 32 {
		n = 32
	}
	p.native.alt_n = C.int(n)
}

func (p Params) nativeForCall() (C.struct_whisper_full_params, func(), error) {
	native := p.native
	if (p.initialPromptSet && strings.ContainsRune(p.initialPrompt, 0)) || (p.vadModelPathSet && strings.ContainsRune(p.vadModelPath, 0)) {
		return native, nil, fmt.Errorf("legacy Whisper prompt/model path contains NUL")
	}
	var prompt, model *C.char
	if p.initialPromptSet {
		prompt = C.CString(p.initialPrompt)
		native.initial_prompt = prompt
	}
	if p.vadModelPathSet {
		model = C.CString(p.vadModelPath)
		native.vad_model_path = model
	}
	return native, func() {
		if prompt != nil {
			C.free(unsafe.Pointer(prompt))
		}
		if model != nil {
			C.free(unsafe.Pointer(model))
		}
	}, nil
}
