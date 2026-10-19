package whisper

/*
#include "crispasr.h"
#include "crispasr_session.h"
static int whisper_go_token_text_length(const char* text) {
    if (!text) return 0;
    for (int n = 0; n <= 1048576; ++n) {
        if (!text[n]) return n;
    }
    return -1;
}
*/
import "C"

import (
	"bytes"
	"errors"
	"unsafe"
)

const legacyMaxTokens = 1_000_000
const legacyMaxText = 1 << 20

// WhisperTokenAlternative owns a candidate ID, raw text and native probability.
// Text is a Go string containing the same bytes as Raw, possibly incomplete UTF-8.
type WhisperTokenAlternative struct {
	ID   Token
	Text string
	Raw  []byte
	P    float32
}

// WhisperTokenSnapshot owns all its text and alternatives. Times are native
// centiseconds, preserving -1 when unavailable. Results survive model close.
// Serialize reads, inference and close on the same native context.
type WhisperTokenSnapshot struct {
	ID           Token
	Text         string
	Raw          []byte
	T0, T1, DTW  int64
	P            float32
	Alternatives []WhisperTokenAlternative
}

func (ctx *Context) legacyTokenCount(segment int) (int, error) {
	if ctx == nil || segment < 0 || uint64(segment) > 2147483647 {
		return 0, errors.New("context and nonnegative int32 segment required")
	}
	segments := ctx.Whisper_full_n_segments()
	if segments < 0 || segment >= segments {
		return 0, errors.New("segment index outside native results")
	}
	count := ctx.Whisper_full_n_tokens(segment)
	if count < 0 || count > legacyMaxTokens {
		return 0, errors.New("invalid or excessive native token count")
	}
	return count, nil
}

// Whisper_token_snapshot copies a token after checking its indices and counts.
// Text is limited to 1 MiB per token/alternative; at most 32 alternatives are valid.
func (ctx *Context) Whisper_token_snapshot(segment, token int) (WhisperTokenSnapshot, error) {
	if token < 0 || uint64(token) > 2147483647 {
		return WhisperTokenSnapshot{}, errors.New("nonnegative int32 token required")
	}
	count, err := ctx.legacyTokenCount(segment)
	if err != nil {
		return WhisperTokenSnapshot{}, err
	}
	if token >= count {
		return WhisperTokenSnapshot{}, errors.New("token index outside native results")
	}
	return ctx.copyLegacyToken(segment, token)
}

// Whisper_token_snapshots copies all tokens in a segment (at most one million).
// Empty lists are valid; malformed native counts/text return an error.
func (ctx *Context) Whisper_token_snapshots(segment int) ([]WhisperTokenSnapshot, error) {
	count, err := ctx.legacyTokenCount(segment)
	if err != nil {
		return nil, err
	}
	result := make([]WhisperTokenSnapshot, 0, count)
	for token := 0; token < count; token++ {
		snapshot, err := ctx.copyLegacyToken(segment, token)
		if err != nil {
			return nil, err
		}
		result = append(result, snapshot)
	}
	return result, nil
}

func (ctx *Context) copyLegacyToken(segment, token int) (WhisperTokenSnapshot, error) {
	native := (*C.struct_whisper_context)(ctx)
	ptr := C.whisper_full_get_token_text(native, C.int(segment), C.int(token))
	length := C.whisper_go_token_text_length(ptr)
	if length < 0 {
		return WhisperTokenSnapshot{}, errors.New("native token text exceeds 1 MiB")
	}
	var raw []byte
	if length != 0 {
		raw = C.GoBytes(unsafe.Pointer(ptr), length)
	} else {
		raw = []byte{}
	}
	count := int(C.crispasr_token_n_alts(native, C.int(segment), C.int(token)))
	if count < 0 || count > 32 {
		return WhisperTokenSnapshot{}, errors.New("invalid native alternative count")
	}
	alternatives := make([]WhisperTokenAlternative, 0, count)
	for alternative := 0; alternative < count; alternative++ {
		text, err := ctx.copyLegacyAlternative(segment, token, alternative)
		if err != nil {
			return WhisperTokenSnapshot{}, err
		}
		alternatives = append(alternatives, WhisperTokenAlternative{
			ID:  Token(C.crispasr_token_alt_id(native, C.int(segment), C.int(token), C.int(alternative))),
			P:   float32(C.crispasr_token_alt_p(native, C.int(segment), C.int(token), C.int(alternative))),
			Raw: text, Text: string(text),
		})
	}
	data := ctx.Whisper_full_get_token_data(segment, token)
	return WhisperTokenSnapshot{ID: data.Id(), Text: string(raw), Raw: raw, T0: data.T0(), T1: data.T1(), DTW: data.TDtw(), P: float32(data.p), Alternatives: alternatives}, nil
}

func (ctx *Context) copyLegacyAlternative(segment, token, alternative int) ([]byte, error) {
	capacity := 256
	for {
		buffer := bytes.Repeat([]byte{0xff}, capacity)
		n := int(C.crispasr_token_alt_text((*C.struct_whisper_context)(ctx), C.int(segment), C.int(token), C.int(alternative), (*C.char)(unsafe.Pointer(&buffer[0])), C.int(capacity)))
		if n == -1 && capacity < legacyMaxText+1 {
			capacity *= 2
			if capacity > legacyMaxText+1 {
				capacity = legacyMaxText + 1
			}
			continue
		}
		if n < 0 || n >= capacity {
			return nil, errors.New("invalid or excessive native alternative text length")
		}
		if buffer[n] != 0 || bytes.IndexByte(buffer[:n], 0) >= 0 {
			return nil, errors.New("malformed native alternative text")
		}
		return buffer[:n:n], nil
	}
}
