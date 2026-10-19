//go:build probe

package whisper

import (
	"bytes"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestProbeTokenRawBytesAfterFree(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	res, err := sess.Transcribe(pcm)
	if err != nil {
		t.Fatalf("Transcribe failed: %v", err)
	}
	if len(res.Segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(res.Segments))
	}
	seg := res.Segments[0]
	if seg.Text != "世" {
		t.Fatalf("segment text mismatch: got %q want %q", seg.Text, "世")
	}
	if seg.T0 != 100 || seg.T1 != 200 {
		t.Fatalf("segment times mismatch: %d/%d", seg.T0, seg.T1)
	}
	if seg.Speaker != "(Speaker 1) " {
		t.Fatalf("speaker mismatch: %q", seg.Speaker)
	}
	if seg.NoSpeechProb != -1.0 {
		t.Fatalf("no_speech_prob should be -1 sentinel, got %f", seg.NoSpeechProb)
	}
	if !seg.HasTimestamps() {
		t.Fatalf("segment should have timestamps")
	}
	if len(seg.Words) != 1 {
		t.Fatalf("expected 1 word, got %d", len(seg.Words))
	}
	w := seg.Words[0]
	if w.Text != "世" {
		t.Fatalf("word text mismatch: %q", w.Text)
	}
	if w.T0 != -1 || w.T1 != -1 {
		t.Fatalf("word times should be -1 sentinel, got %d/%d", w.T0, w.T1)
	}
	if w.HasTimestamps() {
		t.Fatalf("word with sentinel should not have timestamps")
	}
	if w.P != 0.9 {
		t.Fatalf("word p mismatch: %f", w.P)
	}
	if len(w.Alternatives) != 1 {
		t.Fatalf("expected 1 alternative, got %d", len(w.Alternatives))
	}
	if w.Alternatives[0].Text != "alternative" || w.Alternatives[0].P != 0.25 {
		t.Fatalf("alternative mismatch: %+v", w.Alternatives[0])
	}
	if len(seg.Tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(seg.Tokens))
	}
	tok0 := seg.Tokens[0]
	tok1 := seg.Tokens[1]
	if !bytes.Equal(tok0.Raw, []byte{0xe6, 0x97}) {
		t.Fatalf("tok0 raw mismatch: %v", tok0.Raw)
	}
	if !bytes.Equal(tok1.Raw, []byte{0xa5}) {
		t.Fatalf("tok1 raw mismatch: %v", tok1.Raw)
	}
	combined := append([]byte{}, tok0.Raw...)
	combined = append(combined, tok1.Raw...)
	if string(combined) != "日" {
		t.Fatalf("combined raw should be 日, got %q", string(combined))
	}
	if tok0.T0 != -1 || tok0.T1 != -1 {
		t.Fatalf("tok0 times should be -1 sentinel, got %d/%d", tok0.T0, tok0.T1)
	}
	if tok0.HasTimestamps() {
		t.Fatalf("tok0 with sentinel should not have timestamps")
	}
	if tok1.T0 != 123 || tok1.T1 != 124 {
		t.Fatalf("tok1 times mismatch: %d/%d", tok1.T0, tok1.T1)
	}
	if !tok1.HasTimestamps() {
		t.Fatalf("tok1 should have timestamps")
	}
	if tok1.P != 0.75 {
		t.Fatalf("tok1 p mismatch: %f", tok1.P)
	}
	if probeResultFreesGet() != 1 {
		t.Fatalf("result should have been freed exactly once, got %d", probeResultFreesGet())
	}
	probeMutate()
	if !bytes.Equal(tok0.Raw, []byte{0xe6, 0x97}) {
		t.Fatalf("raw bytes not copied before free/mutate, got %v", tok0.Raw)
	}
}

func TestProbeAllResultReaders(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	entryPoints := []struct {
		name string
		fn   func() (*TranscribeResult, error)
	}{
		{"Transcribe", func() (*TranscribeResult, error) { return sess.Transcribe(pcm) }},
		{"TranscribeLang", func() (*TranscribeResult, error) { return sess.TranscribeLang(pcm, "en") }},
		{"TranscribeChunked", func() (*TranscribeResult, error) { return sess.TranscribeChunked(pcm, 30, 1, "") }},
		{"TranscribeVAD", func() (*TranscribeResult, error) { return sess.TranscribeVAD(pcm, 16000, "vad.gguf") }},
		{"TranscribeWithLogits", func() (*TranscribeResult, error) { r, _, err := sess.TranscribeWithLogits(pcm); return r, err }},
	}
	for _, ep := range entryPoints {
		probeReset()
		probeModeSet(0)
		res, err := ep.fn()
		if err != nil {
			t.Fatalf("%s failed: %v", ep.name, err)
		}
		if len(res.Segments) == 0 {
			t.Fatalf("%s should return segments", ep.name)
		}
		seg := res.Segments[0]
		if seg.Text == "" {
			t.Fatalf("%s segment text empty", ep.name)
		}
		if len(seg.Tokens) != 2 {
			t.Fatalf("%s tokens not preserved: %d", ep.name, len(seg.Tokens))
		}
		if len(seg.Words) == 0 || len(seg.Words[0].Alternatives) == 0 {
			t.Fatalf("%s alternatives not preserved", ep.name)
		}
		if seg.Speaker == "" {
			t.Fatalf("%s speaker not preserved", ep.name)
		}
	}
}

func TestProbeEmptyResult(t *testing.T) {
	probeReset()
	probeModeSet(1)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	res, err := sess.Transcribe(pcm)
	if err != nil {
		t.Fatalf("empty Transcribe should not error: %v", err)
	}
	if len(res.Segments) != 0 {
		t.Fatalf("empty result should have 0 segments, got %d", len(res.Segments))
	}
	pianoRes, err := sess.Piano(pcm)
	if err != nil {
		t.Fatalf("Piano empty should not error: %v", err)
	}
	if len(pianoRes.Notes) != 0 || len(pianoRes.Programs) != 0 {
		t.Fatalf("Piano empty should have no notes, got %d/%d", len(pianoRes.Notes), len(pianoRes.Programs))
	}
	chRes, err := sess.Chords(pcm, 16000)
	if err != nil {
		t.Fatalf("Chords empty should not error: %v", err)
	}
	if len(chRes.Spans) != 0 {
		t.Fatalf("Chords empty should have no spans")
	}
	beatRes, err := sess.Beats(pcm, 22050)
	if err != nil {
		t.Fatalf("Beats empty should not error: %v", err)
	}
	if len(beatRes.Events) != 0 {
		t.Fatalf("Beats empty should have no events")
	}
	tabRes, err := sess.Tab(pcm, 16000)
	if err != nil {
		t.Fatalf("Tab empty should not error: %v", err)
	}
	if tabRes.Frames != 0 || len(tabRes.Emissions) != 0 {
		t.Fatalf("Tab empty should have 0 frames, got %d", tabRes.Frames)
	}
}

func TestProbeErrorResult(t *testing.T) {
	probeReset()
	probeModeSet(2)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	if _, err := sess.Transcribe(pcm); err == nil {
		t.Fatalf("Transcribe should error in mode 2")
	}
	if _, err := sess.Piano(pcm); err == nil {
		t.Fatalf("Piano should error in mode 2")
	}
	if _, err := sess.Chords(pcm, 16000); err == nil {
		t.Fatalf("Chords should error in mode 2")
	}
	if _, err := sess.Beats(pcm, 22050); err == nil {
		t.Fatalf("Beats should error in mode 2")
	}
	if _, err := sess.Tab(pcm, 16000); err == nil {
		t.Fatalf("Tab should error in mode 2")
	}
	probeModeSet(0)
	if _, err := sess.Beats(pcm, 16000); err == nil {
		t.Fatalf("Beats with mismatched rate 16000 should error")
	}
}

func TestProbeReplayNoiseRejectedBeforeNative(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	nFrames := 3
	contentDim := sess.ConvertContentDim()
	if contentDim != 2 {
		t.Fatalf("content dim should be 2, got %d", contentDim)
	}
	content := make([]float32, nFrames*contentDim)
	content[1] = 1
	f0 := make([]float32, nFrames)
	f0[2] = 5
	interCh := sess.ConvertInterChannels()
	upsample := sess.ConvertUpsampleProduct()
	if interCh != 2 || upsample != 4 {
		t.Fatalf("interCh/upsample mismatch: %d/%d", interCh, upsample)
	}
	noiseZp := make([]float32, interCh*nFrames)
	noiseZp[5] = 5
	noiseSine := make([]float32, nFrames*upsample)
	noiseSine[11] = 11
	callsBefore := probeCallsGet()
	if _, err := sess.Convert(content, nFrames, f0, 0, noiseZp, noiseSine); err != nil {
		t.Fatalf("Convert with correct noise should succeed: %v", err)
	}
	if probeCallsGet() != callsBefore+1 {
		t.Fatalf("Convert should have called native once")
	}
	probeReset()
	shortZp := make([]float32, 2)
	if _, err := sess.Convert(content, nFrames, f0, 0, shortZp, noiseSine); err == nil {
		t.Fatalf("Convert with short noiseZp should error")
	}
	if probeCallsGet() != 0 {
		t.Fatalf("short noiseZp should be rejected BEFORE native, calls=%d", probeCallsGet())
	}
	shortSine := make([]float32, 2)
	if _, err := sess.Convert(content, nFrames, f0, 0, noiseZp, shortSine); err == nil {
		t.Fatalf("Convert with short noiseSine should error")
	}
	if probeCallsGet() != 0 {
		t.Fatalf("short noiseSine should be rejected BEFORE native")
	}
	if _, err := sess.Convert([]float32{1, 2}, 1000000000, f0, 0, nil, nil); err == nil {
		t.Fatalf("Convert with huge nFrames should error")
	}
}

func TestProbeAllocatedPCMFrees(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	liveBefore := probeLiveGet()
	pcm, err := sess.GetDisclaimerPCM()
	if err != nil {
		t.Fatalf("GetDisclaimerPCM failed: %v", err)
	}
	if len(pcm) != 2 || pcm[0] != 0.25 || pcm[1] != 0.5 {
		t.Fatalf("disclaimer pcm mismatch: %v", pcm)
	}
	if probeLiveGet() != liveBefore {
		t.Fatalf("GetDisclaimerPCM should free allocated PCM, live before %d after %d", liveBefore, probeLiveGet())
	}
	liveBefore = probeLiveGet()
	pcm2, err := sess.Synthesize("hello")
	if err != nil {
		t.Fatalf("Synthesize failed: %v", err)
	}
	if len(pcm2) != 2 {
		t.Fatalf("Synthesize pcm len mismatch")
	}
	if probeLiveGet() != liveBefore {
		t.Fatalf("Synthesize should free PCM, leak detected")
	}
	probeModeSet(1)
	if _, err := sess.GetDisclaimerPCM(); err == nil {
		t.Fatalf("GetDisclaimerPCM with 0 should error")
	}
	if probeLiveGet() != liveBefore {
		t.Fatalf("error case should not leak, live %d", probeLiveGet())
	}
}

func TestProbeSynchronousStreaming(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	var chunks [][]float32
	var finals []bool
	err := sess.SynthesizeStreaming("hello", func(pcm []float32, isFinal bool) {
		cp := make([]float32, len(pcm))
		copy(cp, pcm)
		chunks = append(chunks, cp)
		finals = append(finals, isFinal)
		if len(pcm) > 0 {
			pcm[0] = 999
		}
	})
	if err != nil {
		t.Fatalf("SynthesizeStreaming failed: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 synchronous chunks, got %d", len(chunks))
	}
	if chunks[0][0] != 0.25 || chunks[0][1] != 0.5 {
		t.Fatalf("first chunk mismatch: %v", chunks[0])
	}
	if chunks[1][0] != 0.75 || chunks[1][1] != 1.0 {
		t.Fatalf("second chunk mismatch: %v", chunks[1])
	}
	if finals[0] != false || finals[1] != true {
		t.Fatalf("final flags mismatch: %v", finals)
	}
	if chunks[0][0] == 999 {
		t.Fatalf("wrapper did not copy PCM before callback, mutation leaked")
	}
}

func TestProbeStreamingPanicRecovery(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	err := sess.SynthesizeStreaming("hello", func(pcm []float32, isFinal bool) {
		panic("test panic")
	})
	if err == nil {
		t.Fatalf("panic in callback should be recovered as error")
	}
	probeModeSet(0)
	var chunks int
	err = sess.SynthesizeStreaming("hello", func(pcm []float32, isFinal bool) {
		chunks++
	})
	if err != nil {
		t.Fatalf("session should be usable after panic: %v", err)
	}
	if chunks != 2 {
		t.Fatalf("expected 2 chunks after recovery, got %d", chunks)
	}
}

func TestProbeMusicOperations(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	piano, err := sess.Piano(pcm)
	if err != nil {
		t.Fatalf("Piano failed: %v", err)
	}
	if len(piano.Notes) != 2 || len(piano.Programs) != 2 {
		t.Fatalf("Piano counts mismatch: %d/%d", len(piano.Notes), len(piano.Programs))
	}
	if piano.SampleRate != 16000 {
		t.Fatalf("Piano sample rate mismatch: %d", piano.SampleRate)
	}
	if piano.Notes[0].OnsetMs != 0 || piano.Notes[0].OffsetMs != 1 {
		t.Fatalf("Piano note data mismatch: %+v", piano.Notes[0])
	}
	if piano.Programs[0] != 24 || piano.Programs[1] != 128 {
		t.Fatalf("Piano programs mismatch: %v", piano.Programs)
	}
	chords, err := sess.Chords(pcm, 16000)
	if err != nil {
		t.Fatalf("Chords failed: %v", err)
	}
	if len(chords.Spans) != 2 {
		t.Fatalf("Chords spans mismatch: %d", len(chords.Spans))
	}
	if chords.Spans[0].Name != "C" || chords.Spans[1].Name != "Am" {
		t.Fatalf("Chords names mismatch: %v", chords.Spans)
	}
	if chords.VocabSize != 170 {
		t.Fatalf("Chords vocab mismatch: %d", chords.VocabSize)
	}
	beats, err := sess.Beats(pcm, 22050)
	if err != nil {
		t.Fatalf("Beats failed: %v", err)
	}
	if len(beats.Events) != 2 {
		t.Fatalf("Beats events mismatch: %d", len(beats.Events))
	}
	if beats.SampleRate != 22050 || beats.TempoBPM != 123.5 {
		t.Fatalf("Beats metadata mismatch: %d/%f", beats.SampleRate, beats.TempoBPM)
	}
	tab, err := sess.Tab(pcm, 16000)
	if err != nil {
		t.Fatalf("Tab failed: %v", err)
	}
	if tab.Frames != 2 || tab.Strings != 2 || tab.Classes != 3 {
		t.Fatalf("Tab dims mismatch: %d/%d/%d", tab.Frames, tab.Strings, tab.Classes)
	}
	if len(tab.Emissions) != 12 {
		t.Fatalf("Tab emissions len mismatch: %d", len(tab.Emissions))
	}
	if tab.SilentClass != 2 || tab.FramePeriod != 0.02 {
		t.Fatalf("Tab metadata mismatch")
	}
	if len(tab.OpenMidi) != 2 || tab.OpenMidi[0] != 40 {
		t.Fatalf("Tab openMidi mismatch: %v", tab.OpenMidi)
	}
	pitch, err := sess.Pitch(pcm, 10.0)
	if err != nil {
		t.Fatalf("Pitch failed: %v", err)
	}
	if len(pitch) != 2 {
		t.Fatalf("Pitch frames mismatch: %d", len(pitch))
	}
	content := make([]float32, 6)
	content[1] = 1
	f0 := make([]float32, 3)
	f0[2] = 5
	noiseZp := make([]float32, 6)
	noiseZp[5] = 5
	noiseSine := make([]float32, 12)
	noiseSine[11] = 11
	audio, err := sess.Convert(content, 3, f0, 0, noiseZp, noiseSine)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}
	if len(audio) != 6 || audio[0] != 0 || audio[5] != 5 {
		t.Fatalf("Convert audio mismatch: %v", audio)
	}
}

func TestProbeConvertValidation(t *testing.T) {
	probeReset()
	sess := newProbeSession(t)
	if _, err := sess.Convert(nil, 0, nil, 0, nil, nil); err == nil {
		t.Fatalf("Convert with nFrames 0 should error")
	}
	content := make([]float32, 6)
	content[1] = 1
	f0 := make([]float32, 3)
	f0[2] = 5
	if _, err := sess.Convert(content, 3, f0, 5, nil, nil); err == nil {
		t.Fatalf("Convert with invalid speaker should error")
	}
	if _, err := sess.Convert(content, 1<<30, f0, 0, nil, nil); err == nil {
		t.Fatalf("Convert with huge nFrames should error")
	}
}

func TestProbeBeatNativeRateMismatch(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	if _, err := sess.Beats(pcm, 22050); err != nil {
		t.Fatalf("Beats with correct rate should succeed: %v", err)
	}
	if _, err := sess.Beats(pcm, 16000); err == nil {
		t.Fatalf("Beats with mismatched rate 16000 should error")
	}
	if _, err := sess.Beats(nil, 22050); err == nil {
		t.Fatalf("Beats with empty PCM should error before native")
	}
}

func TestProbeTabOverflow(t *testing.T) {
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	tab, err := sess.Tab(pcm, 16000)
	if err != nil {
		t.Fatalf("Tab failed: %v", err)
	}
	if len(tab.Emissions) != int(tab.Frames)*int(tab.Strings)*int(tab.Classes) {
		t.Fatalf("Tab emissions length should match product")
	}
}

func TestProbeHasTimestamps(t *testing.T) {
	seg := TranscribeSegment{T0: 100, T1: 200}
	if !seg.HasTimestamps() {
		t.Fatalf("valid segment should have timestamps")
	}
	seg2 := TranscribeSegment{T0: -1, T1: -1}
	if seg2.HasTimestamps() {
		t.Fatalf("sentinel segment should not have timestamps")
	}
	word := TranscribeWord{T0: 50, T1: 100}
	if !word.HasTimestamps() {
		t.Fatalf("valid word should have timestamps")
	}
	tok := SessionToken{T0: 123, T1: 124}
	if !tok.HasTimestamps() {
		t.Fatalf("valid token should have timestamps")
	}
	tok2 := SessionToken{T0: -1, T1: -1}
	if tok2.HasTimestamps() {
		t.Fatalf("sentinel token should not have timestamps")
	}
}

func TestProbeDisclaimerAndLastError(t *testing.T) {
	txt := DisclaimerText()
	if txt != "AI generated" {
		t.Fatalf("DisclaimerText should be 'AI generated', got %q", txt)
	}
	probeModeSet(0)
	sess := newProbeSession(t)
	errStr := sess.LastSynthError()
	if errStr != "probe synthesis failure" {
		t.Fatalf("LastSynthError mismatch: %q", errStr)
	}
	if err := sess.SetTTSCfgScale(1.5); err != nil {
		t.Fatalf("SetTTSCfgScale with 1.5 should succeed: %v", err)
	}
	if err := sess.SetTTSCfgScale(1.0); err == nil {
		t.Fatalf("SetTTSCfgScale with 1.0 should fail")
	}
}

func TestProbeSynthesisStreamingCallbackLifetime(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	var firstCopy []float32
	err := sess.SynthesizeStreaming("test", func(pcm []float32, isFinal bool) {
		if !isFinal {
			firstCopy = make([]float32, len(pcm))
			copy(firstCopy, pcm)
		}
		for i := range pcm {
			pcm[i] = 0
		}
	})
	if err != nil {
		t.Fatalf("SynthesizeStreaming failed: %v", err)
	}
	if len(firstCopy) != 2 || firstCopy[0] != 0.25 {
		t.Fatalf("callback lifetime: first chunk not preserved, got %v", firstCopy)
	}
	if err := sess.SynthesizeStreaming("test", nil); err == nil {
		t.Fatalf("nil callback should error")
	}
}

func TestProbeSetVoiceSamples(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	if err := sess.SetVoiceSamples(pcm, 24000, "hello"); err != nil {
		t.Fatalf("SetVoiceSamples valid should succeed: %v", err)
	}
	if err := sess.SetVoiceSamples(pcm, 16000, "hello"); err == nil {
		t.Fatalf("SetVoiceSamples with wrong rate should fail")
	}
	if err := sess.SetVoiceSamples(nil, 24000, "hello"); err == nil {
		t.Fatalf("SetVoiceSamples with empty PCM should fail before native")
	}
}

func TestProbeAudioHelpers(t *testing.T) {
	if _, err := PcmToWav(nil, 16000); err == nil {
		t.Fatalf("PcmToWav with empty PCM should error before native")
	}
	if _, err := PcmToWav([]float32{1, 2}, 0); err == nil {
		t.Fatalf("PcmToWav with invalid rate should error")
	}
}

func TestProbeCopyBeforeFree(t *testing.T) {
	probeReset()
	probeModeSet(0)
	sess := newProbeSession(t)
	pcm := []float32{1, 3, 5}
	res, err := sess.Transcribe(pcm)
	if err != nil {
		t.Fatalf("Transcribe failed: %v", err)
	}
	seg := res.Segments[0]
	origRaw := make([]byte, len(seg.Tokens[0].Raw))
	copy(origRaw, seg.Tokens[0].Raw)
	probeMutate()
	if !bytes.Equal(seg.Tokens[0].Raw, origRaw) {
		t.Fatalf("token raw not copied before free, mutated to %v", seg.Tokens[0].Raw)
	}
	origAlt := seg.Words[0].Alternatives[0].Text
	probeMutate()
	if seg.Words[0].Alternatives[0].Text != origAlt {
		t.Fatalf("alternative not copied")
	}
	origSpeaker := seg.Speaker
	probeMutate()
	if seg.Speaker != origSpeaker {
		t.Fatalf("speaker not copied")
	}
}

func TestProbeKokoroLangHelpers(t *testing.T) {
	// Exercise both true and false branches to catch garbage upper bits on bool return.
	// The probe fixture returns true only for de/de-DE (german) and en/de (native voice).
	if !KokoroLangIsGerman("de") {
		t.Fatalf("KokoroLangIsGerman(de) should be true")
	}
	if !KokoroLangIsGerman("de-DE") {
		t.Fatalf("KokoroLangIsGerman(de-DE) should be true")
	}
	if KokoroLangIsGerman("en") {
		t.Fatalf("KokoroLangIsGerman(en) should be false")
	}
	if KokoroLangIsGerman("fr") {
		t.Fatalf("KokoroLangIsGerman(fr) should be false")
	}
	if KokoroLangIsGerman("") {
		t.Fatalf("KokoroLangIsGerman(\"\") should be false")
	}
	if !KokoroLangHasNativeVoice("en") {
		t.Fatalf("KokoroLangHasNativeVoice(en) should be true")
	}
	if !KokoroLangHasNativeVoice("de") {
		t.Fatalf("KokoroLangHasNativeVoice(de) should be true")
	}
	if KokoroLangHasNativeVoice("fr") {
		t.Fatalf("KokoroLangHasNativeVoice(fr) should be false")
	}
	if KokoroLangHasNativeVoice("") {
		t.Fatalf("KokoroLangHasNativeVoice(\"\") should be false")
	}
}

func TestProbeAudioAllocationErrorsAndAliases(t *testing.T) {
	probeUtilityMode(0)
	audio, rate, err := AudioLoad("probe", 24000)
	if err != nil || rate != 24000 || audio[0] != .25 {
		t.Fatalf("audio load transport: %v %v %v", audio, rate, err)
	}
	probeUtilityMode(3)
	left, right, _, _, channels, err := AudioLoadStereo("probe")
	if err != nil || channels != 1 {
		t.Fatalf("stereo alias transport: %v", err)
	}
	left[0] = 999
	if right[0] != .25 {
		t.Fatal("stereo outputs alias each other")
	}
	pcm := []float32{1, 3, 5}
	out, err := AudioResample(pcm, 16000, 24000)
	if err != nil || out[0] != .25 {
		t.Fatalf("resample transport: %v", err)
	}
	wav, err := PcmToWav(pcm, 16000)
	if err != nil || !bytes.Equal(wav, []byte{87, 0, 65, 86}) {
		t.Fatalf("binary WAV transport: %v %v", wav, err)
	}
	signed, err := C2paSign([]byte{1, 0, 2}, "audio/wav", "", "")
	if err != nil || !bytes.Equal(signed, []byte{1, 0, 2}) {
		t.Fatalf("binary signing transport: %v", err)
	}
	probeUtilityMode(2)
	if _, _, err = AudioLoad("probe", 16000); err == nil {
		t.Fatal("load error swallowed")
	}
	if _, _, _, _, _, err = AudioLoadStereo("probe"); err == nil {
		t.Fatal("stereo error swallowed")
	}
	if _, err = AudioResample(pcm, 16000, 24000); err == nil {
		t.Fatal("resample error swallowed")
	}
	if probeUtilityLive() != 0 {
		t.Fatal("native audio allocation leak")
	}
	probeUtilityMode(0)
}
func TestProbeCallbackReentryAndStreamParent(t *testing.T) {
	probeModeSet(0)
	session := newProbeSession(t)
	if err := session.SynthesizeStreaming("test", func(_ []float32, _ bool) { session.Close() }); err == nil {
		t.Fatal("callback closed live model")
	}
	if err := session.SynthesizeStreaming("test", func(_ []float32, _ bool) { session.InputSampleRate() }); err == nil {
		t.Fatal("callback re-entered native getter")
	}
	before := probeStreamCloses()
	stream, err := session.StreamOpen(3000, 10000, 200, "en", false)
	if err != nil {
		t.Fatal(err)
	}
	session.Close()
	if probeStreamCloses() != before+1 {
		t.Fatal("parent did not close child stream")
	}
	stream.Close()
	if _, err = stream.Feed([]float32{1}); err == nil {
		t.Fatal("closed child entered native")
	}
}

func TestProbeTypedAsrCallbacksAllRoutes(t *testing.T) {
	probeReset()
	session := newProbeSession(t)
	defer session.Close()
	pcm := []float32{1, 3, 5}
	calls := 0
	routes := []func() error{
		func() error { _, e := session.Transcribe(pcm); return e },
		func() error { _, e := session.TranscribeLang(pcm, "en"); return e },
		func() error { _, e := session.TranscribeChunked(pcm, 30, 1, ""); return e },
		func() error { _, e := session.TranscribeVAD(pcm, 16000, "vad"); return e },
		func() error { _, _, e := session.TranscribeWithLogits(pcm); return e },
		func() error { _, e := session.TranscribeVadLang(pcm, 16000, "vad", "en"); return e },
	}
	session.SetProgressCallback(func(processed, total int) {
		if processed != 3 || total != 6 {
			panic("progress arguments")
		}
		calls++
	})
	session.SetSegmentCallback(func(text string, t0, t1 int64, index int) {
		if text != "日本語" || t0 != 123 || t1 != 456 || index != 7 {
			panic("segment arguments")
		}
		calls++
	})
	var raw []byte
	session.SetTokenCallback(func(bytes []byte, index int) {
		if index != 9 {
			panic("index")
		}
		raw = bytes
		calls++
	})
	for _, run := range routes {
		if err := run(); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 18 || !bytes.Equal(raw, []byte{0xe6, 0x97}) {
		t.Fatalf("callback transport %d %v", calls, raw)
	}
	session.SetProgressCallback(nil)
	session.SetSegmentCallback(nil)
	session.SetTokenCallback(nil)
	if err := routes[0](); err != nil || calls != 18 {
		t.Fatal("cleared callbacks fired", err)
	}
	for _, kind := range []string{"progress", "segment", "token"} {
		if kind == "progress" {
			session.SetProgressCallback(func(int, int) { panic("callback sentinel") })
		}
		if kind == "segment" {
			session.SetSegmentCallback(func(string, int64, int64, int) { session.InputSampleRate() })
		}
		if kind == "token" {
			session.SetTokenCallback(func([]byte, int) { session.Close() })
		}
		for _, run := range routes {
			before := probeResultFreesGet()
			if err := run(); err == nil {
				t.Fatal("callback panic escaped error transport", kind)
			}
			if probeResultFreesGet() != before+1 {
				t.Fatal("callback failure leaked result", kind)
			}
		}
		session.SetProgressCallback(nil)
		session.SetSegmentCallback(nil)
		session.SetTokenCallback(nil)
		if err := routes[0](); err != nil {
			t.Fatal("stale callback error", err)
		}
	}
}

func TestProbeParakeetIntegerOffset(t *testing.T) {
	ctx, err := ParakeetInit("probe", 4, true)
	if err != nil {
		t.Fatal(err)
	}
	pcm := []float32{1, 3, 5}
	result, err := ctx.TranscribeAt(pcm, 123)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "日本語" || result.Words[0].T0 != 223 || result.Tokens[0].T0 != -1 {
		t.Fatalf("bad Parakeet offset transport: %#v", result)
	}
	if result.Tokens[0].Text+result.Tokens[1].Text != "日" {
		t.Fatal("legacy token bytes changed")
	}
	if _, err = ctx.Transcribe(pcm, "de"); err == nil {
		t.Fatal("accepted nonexistent language parameter")
	}
	ctx.Close()
	ctx.Close()
	if _, err = ctx.Transcribe(pcm, ""); err == nil {
		t.Fatal("closed context entered native")
	}
}

func TestProbeTitaNetFloatCountAndClosedContext(t *testing.T) {
	probeTitaNetMode(0)
	ctx, err := TitaNetInit("probe", 4)
	if err != nil {
		t.Fatal(err)
	}
	pcm := []float32{1, 3, 5}
	out, err := ctx.Embed(pcm)
	if err != nil || len(out) != 192 || out[0] != 1 {
		t.Fatalf("TitaNet transport: %v %v", out, err)
	}
	for _, mode := range []int{2, 3, 4} {
		probeTitaNetMode(mode)
		if _, err = ctx.Embed(pcm); err == nil {
			t.Fatal("bad native count accepted", mode)
		}
	}
	probeTitaNetMode(0)
	if _, err = ctx.Embed(nil); err == nil {
		t.Fatal("empty PCM accepted")
	}
	ctx.Close()
	ctx.Close()
	if probeTitaNetLive() != 0 {
		t.Fatal("TitaNet leaked")
	}
	if _, err = ctx.Embed(pcm); err == nil {
		t.Fatal("closed context accepted")
	}
	if TitaNetCosineSim(nil, nil) != 0 {
		t.Fatal("empty cosine")
	}
}

// A count is not a byte length. The Linux fixture ends exactly at a protected page.
func TestProbeDrainStreamedTokensExactBufferAndRawFragments(t *testing.T) {
	defer probeModeSet(0)
	probeModeSet(5)
	tokens, err := DrainStreamedTokens()
	if err != nil || len(tokens) != 3 {
		t.Fatalf("drain: %q %v", tokens, err)
	}
	if !bytes.Equal([]byte(tokens[0]), []byte{0xe6, 0x97}) || !bytes.Equal([]byte(tokens[1]), []byte{0xa5}) || tokens[2] != "" {
		t.Fatalf("raw fragments or empty token lost: %q", tokens)
	}
	probeModeSet(0)
	normal, err := DrainStreamedTokens()
	if err != nil || len(normal) != 2 || normal[0] != "tok1" || tokens[2] != "" {
		t.Fatalf("independent drain: %q %v", normal, err)
	}
	probeModeSet(1)
	empty, err := DrainStreamedTokens()
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty drain: %q %v", empty, err)
	}
	for _, mode := range []int{2, 3} {
		probeModeSet(mode)
		if _, err := DrainStreamedTokens(); err == nil {
			t.Fatalf("invalid mode %d accepted", mode)
		}
	}
}

func TestProbeSpeakerDatabaseRosterScoresEnrollmentAndLifetime(t *testing.T) {
	probeDbMode(0)
	defer probeDbMode(0)
	emb := []float32{1, 3, 5}
	before := probeDbCalls()
	if _, err := SpeakerDBOpen("db", "Alice,Bob", false); err == nil {
		t.Fatal("consent missing")
	}
	if _, err := SpeakerDBOpen("db", "", true); err == nil {
		t.Fatal("roster missing")
	}
	if probeDbCalls() != before {
		t.Fatal("invalid open reached native")
	}
	db, err := SpeakerDBOpen("db", "Alice,Bob", true)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if db.Count() != 2 || probeDbLive() != 1 {
		t.Fatal("count/lifetime")
	}
	name, score := db.Match(emb, .5)
	if name != "アリス" || score != .875 {
		t.Fatalf("match: %q %v", name, score)
	}
	probeDbMode(1)
	if name, score := db.Match(emb, -1); name != "" || score != -1 {
		t.Fatal("negative score became match")
	}
	probeDbMode(0)
	if name, score := db.Match(nil, .5); name != "" || score != -1 {
		t.Fatal("empty embedding accepted")
	}
	if err := db.Enroll("Alice", emb, false); err == nil {
		t.Fatal("consent missing")
	}
	if err := db.Enroll("Alice", nil, true); err == nil {
		t.Fatal("empty enroll accepted")
	}
	if err := db.Enroll("Alice", emb, true); err != nil {
		t.Fatal(err)
	}
	probeDbMode(3)
	if err := db.Enroll("Alice", emb, true); err == nil {
		t.Fatal("positive failure accepted")
	}
	probeDbMode(2)
	if _, err := SpeakerDBOpen("db", "Alice,Bob", true); err == nil {
		t.Fatal("null open accepted")
	}
	db.Close()
	db.Close()
	before = probeDbCalls()
	if probeDbLive() != 0 || db.Count() != 0 {
		t.Fatal("closed DB live")
	}
	if name, score := db.Match(emb, .5); name != "" || score != -1 {
		t.Fatal("closed match accepted")
	}
	if err := db.Enroll("Alice", emb, true); err == nil {
		t.Fatal("closed enroll accepted")
	}
	if probeDbCalls() != before {
		t.Fatal("closed call reached native")
	}
}

func TestProbeSessionOpenOptionsAndCapabilityQueries(t *testing.T) {
	probeModeSet(0)
	defer probeModeSet(0)
	opts := SessionOptions{NThreads: 7, UseGPU: false, Verbosity: 2, FlashAttention: false, GPULayers: 3}
	sess, err := SessionOpenWithOptions("model", "probe", opts)
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
	probeModeSet(1)
	sess, err = SessionOpenWithOptions("model", "", DefaultSessionOptions())
	if err != nil {
		t.Fatal(err)
	}
	sess.Close()
	probeModeSet(2)
	if _, err = SessionOpenWithOptions("model", "probe", opts); err == nil {
		t.Fatal("native null accepted")
	}
	probeModeSet(0)
	opts.GPULayers = -2
	if _, err = SessionOpenWithOptions("model", "probe", opts); err == nil {
		t.Fatal("invalid GPU layers")
	}
	if _, err = SessionOpenWithOptions("model\x00hidden", "", DefaultSessionOptions()); err == nil {
		t.Fatal("NUL path")
	}
	verbs, err := BackendCapabilities("probe")
	if err != nil || len(verbs) != 2 || verbs[0] != "asr" || verbs[1] != "tts" {
		t.Fatalf("capabilities: %q %v", verbs, err)
	}
	verbs, err = BackendCapabilities("long")
	if err != nil || len(verbs) != 1 || len(verbs[0]) != 600 {
		t.Fatalf("capability growth: %q %v", verbs, err)
	}
	if _, err = BackendCapabilities("unknown"); err == nil {
		t.Fatal("unknown backend accepted")
	}
	all, err := ListBackendCapabilities()
	if err != nil || len(all["probe"]) != 2 {
		t.Fatalf("all capabilities: %#v %v", all, err)
	}
}
func TestProbeScoreTextsTransportAndPositiveErrors(t *testing.T) {
	probeModeSet(0)
	defer probeModeSet(0)
	sess := newProbeSession(t)
	scores, counts, err := sess.ScoreTexts([]float32{1, 3, 5}, "ja", "prompt", []string{"one", "two words"})
	if err != nil || len(scores) != 2 || scores[1] != -.75 || counts[1] != 3 {
		t.Fatalf("scores: %v %v %v", scores, counts, err)
	}
	probeModeSet(2)
	if _, _, err = sess.ScoreTexts([]float32{1, 3, 5}, "ja", "prompt", []string{"one", "two words"}); err == nil {
		t.Fatal("positive native error accepted")
	}
}

func TestProbeStandaloneTextProcessorsOwnedOutputsAndClose(t *testing.T) {
	probeTextMode(0)
	defer probeTextMode(0)
	type processor interface {
		Process(string) (string, error)
		Close()
	}
	factories := []struct {
		open     func(string) (processor, error)
		expected string
	}{
		{func(path string) (processor, error) { return TruecaseInit(path) }, "Grüße"},
		{func(path string) (processor, error) { return PCSInit(path) }, "Hello."},
		{func(path string) (processor, error) { return PuncInit(path) }, "Hello!"},
	}
	for _, factory := range factories {
		model, err := factory.open("model")
		if err != nil {
			t.Fatal(err)
		}
		out, err := model.Process("hello")
		if err != nil || out != factory.expected || probeTextLive() != 0 {
			t.Fatalf("text: %q %v live=%d", out, err, probeTextLive())
		}
		probeTextMode(2)
		empty, err := model.Process("")
		if err != nil || empty != "" || probeTextLive() != 0 {
			t.Fatalf("empty: %q %v", empty, err)
		}
		probeTextMode(1)
		if _, err = model.Process("hello"); err == nil {
			t.Fatal("native null accepted")
		}
		probeTextMode(0)
		if _, err = model.Process("bad\x00text"); err == nil {
			t.Fatal("NUL accepted")
		}
		model.Close()
		model.Close()
		if probeTextContexts() != 0 || out != factory.expected {
			t.Fatal("context/output lifetime")
		}
		if _, err = model.Process("hello"); err == nil {
			t.Fatal("closed processor accepted")
		}
		probeTextMode(3)
		if _, err = factory.open("model"); err == nil {
			t.Fatal("native init failure accepted")
		}
		probeTextMode(0)
	}
}

func TestProbeNativeContainerZeroAndOverflowCountsAlwaysFree(t *testing.T) {
	defer probeUtilityMode(0)
	for _, mode := range []int{5, 6} {
		probeUtilityMode(mode)
		if _, err := PcmToWav([]float32{1, 3, 5}, 16000); err == nil {
			t.Fatal("invalid WAV size accepted")
		}
		if probeUtilityLive() != 0 {
			t.Fatalf("WAV leaked on invalid size mode %d", mode)
		}
		if _, err := C2paSign([]byte{1, 0, 2}, "audio/wav", "", ""); err == nil {
			t.Fatal("invalid signed size accepted")
		}
		if probeUtilityLive() != 0 {
			t.Fatalf("signed container leaked on invalid size mode %d", mode)
		}
	}
}

func TestProbeMissingWordConfidenceRemainsUnavailable(t *testing.T) {
	probeModeSet(10)
	defer probeModeSet(0)
	sess := newProbeSession(t)
	result, err := sess.Transcribe([]float32{1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if result.Segments[0].Words[0].P != -1 {
		t.Fatalf("unavailable confidence fabricated as %v", result.Segments[0].Words[0].P)
	}
}

func TestProbePolledSegmentsCopyMetadataAndRejectInvalidCounts(t *testing.T) {
	probeModeSet(0)
	defer probeModeSet(0)
	probeStreamedReady(1)
	before := probeResultFreesGet()
	if GetStreamedSegmentCount() != 1 {
		t.Fatal("queued count")
	}
	result, err := DrainStreamedSegments()
	if err != nil || len(result.Segments) != 1 || result.Segments[0].Speaker != "(Speaker 1) " || len(result.Segments[0].Words[0].Alternatives) != 1 || !bytes.Equal(result.Segments[0].Tokens[0].Raw, []byte{0xe6, 0x97}) {
		t.Fatalf("polled metadata: %#v %v", result, err)
	}
	if probeResultFreesGet() != before+1 || GetStreamedSegmentCount() != 0 {
		t.Fatal("polled result not freed/drained")
	}
	if empty, err := DrainStreamedSegments(); err != nil || empty != nil {
		t.Fatal("empty drain")
	}
	for _, mode := range []int{11, 12, 13, 14} {
		probeModeSet(mode)
		probeStreamedReady(1)
		before = probeResultFreesGet()
		if _, err := DrainStreamedSegments(); err == nil {
			t.Fatalf("invalid count mode %d accepted", mode)
		}
		if probeResultFreesGet() != before+1 {
			t.Fatal("invalid result leaked")
		}
	}
	sess := newProbeSession(t)
	for _, mode := range []int{15, 16} {
		probeModeSet(mode)
		before = probeResultFreesGet()
		if _, _, err := sess.TranscribeWithLogits([]float32{1, 3, 5}); err == nil {
			t.Fatal("invalid logits accepted")
		}
		if probeResultFreesGet() != before+1 {
			t.Fatal("invalid logits leaked")
		}
	}
}

func TestProbeLegacyEmptyPCMReturnsError(t *testing.T) {
	var ctx *Context
	if err := ctx.Whisper_full(Params{}, nil, nil, nil, nil); err == nil {
		t.Fatal("nil context/empty PCM accepted")
	}
}

func TestProbeLegacyParamsCopiesStringsControlsAndFailures(t *testing.T) {
	ctx := newProbeLegacyContext()
	p := ctx.Whisper_full_default_params(SAMPLING_GREEDY)
	p.SetThreads(3)
	p.SetBestOf(7)
	p.SetBestOf(0)
	p.SetDetectLanguage(true)
	p.SetSuppressBlank(false)
	p.SetTdrz(true)
	p.SetAltN(100)
	p.SetInitialPrompt("old 日本語")
	p.SetVADModelPath("old-vad")
	previous := p
	p.SetInitialPrompt("new 日本語")
	p.SetVADModelPath("new-vad")
	pcm := []float32{1, 3, 5}
	encoded, segments, progress := 0, 0, 0
	probeLegacyMode(0)
	run := func(options Params, prompt, model string) {
		t.Helper()
		probeLegacyExpect(prompt, model)
		if err := ctx.Whisper_full(options, pcm, func() bool { encoded++; runtime.GC(); return true }, func(int) { segments++ }, func(int) { progress++ }); err != nil {
			t.Fatal(err)
		}
	}
	run(previous, "old 日本語", "old-vad")
	run(p, "new 日本語", "new-vad")
	if encoded != 2 || segments != 2 || progress != 2 {
		t.Fatalf("callbacks %d/%d/%d", encoded, segments, progress)
	}
	if !strings.Contains(previous.String(), "initial_prompt=old 日本語") || !strings.Contains(p.String(), "initial_prompt=new 日本語") {
		t.Fatal("parameter copies share prompts")
	}
	for i := 0; i < 64; i++ {
		p.SetInitialPrompt("replacement")
		p.SetVADModelPath("replacement-vad")
		run(p, "replacement", "replacement-vad")
	}
	probeLegacyExpect("old 日本語", "old-vad")
	parallelCalls := probeParallelByRefCalls()
	if err := ctx.Whisper_full_parallel(previous, pcm, 2, func() bool { return true }, func(int) {}); err != nil {
		t.Fatal(err)
	}
	if probeParallelByRefCalls() != parallelCalls+1 {
		t.Fatal("parallel decode bypassed the pointer/progress wrapper")
	}
	parallelProgress, parallelSegments, parallelEncoded := 0, 0, 0
	runParallel := func() error {
		return ctx.Whisper_full_parallel_with_progress(previous, pcm, 2,
			func() bool { parallelEncoded++; runtime.GC(); return true },
			func(n int) { parallelSegments += n }, func(n int) { parallelProgress += n })
	}
	if err := runParallel(); err != nil {
		t.Fatal(err)
	}
	if parallelProgress != 50 || parallelSegments != 1 || parallelEncoded != 1 {
		t.Fatalf("parallel callbacks %d/%d/%d", parallelEncoded, parallelSegments, parallelProgress)
	}
	// Returning without a callback must not retain the previous closures.
	if err := ctx.Whisper_full_parallel(previous, pcm, 2, nil, nil); err != nil {
		t.Fatal(err)
	}
	if parallelProgress != 50 || parallelSegments != 1 || parallelEncoded != 1 {
		t.Fatal("parallel callbacks retained after return")
	}
	probeLegacyMode(1)
	if err := runParallel(); err == nil {
		t.Fatal("parallel progress route ignored native failure")
	}
	if err := ctx.Whisper_full(previous, pcm, nil, nil, nil); err == nil {
		t.Fatal("native failure ignored")
	}
	if err := ctx.Whisper_full_parallel(previous, pcm, 2, nil, nil); err == nil {
		t.Fatal("parallel failure ignored")
	}
	probeLegacyMode(0)
	if err := ctx.Whisper_full_parallel(previous, pcm, 2, nil, nil); err != nil {
		t.Fatal(err)
	}
	if parallelProgress != 50 || parallelSegments != 1 || parallelEncoded != 1 {
		t.Fatal("parallel callbacks retained after failure")
	}
	calls := probeLegacyCalls()
	if err := ctx.Whisper_full(previous, nil, nil, nil, nil); err == nil {
		t.Fatal("empty PCM accepted")
	}
	if err := ctx.Whisper_full_parallel(previous, pcm, 0, nil, nil); err == nil {
		t.Fatal("zero processors accepted")
	}
	invalid := previous
	invalid.SetInitialPrompt("bad\x00prompt")
	if err := ctx.Whisper_full(invalid, pcm, nil, nil, nil); err == nil {
		t.Fatal("NUL prompt accepted")
	}
	invalid = previous
	invalid.SetVADModelPath("bad\x00model")
	if err := ctx.Whisper_full_parallel(invalid, pcm, 2, nil, nil); err == nil {
		t.Fatal("NUL model accepted")
	}
	if probeLegacyCalls() != calls {
		t.Fatal("invalid inputs reached native code")
	}
	previous.SetAltN(-2)
	if previous.native.alt_n != 0 {
		t.Fatal("negative candidate count not clamped")
	}
	previous.SetAltN(5)
	if previous.native.alt_n != 5 {
		t.Fatal("candidate count lost")
	}
}

func TestProbeLegacyDTWConstruction(t *testing.T) {
	probeReset()
	for _, test := range []struct {
		preset      AlignmentHeadsPreset
		native, top int
	}{
		{AHEADS_N_TOP_MOST, 1, 3}, {AHEADS_TINY_EN, 3, 0}, {AHEADS_TINY, 4, 0},
		{AHEADS_BASE_EN, 5, 0}, {AHEADS_BASE, 6, 0}, {AHEADS_SMALL_EN, 7, 0}, {AHEADS_SMALL, 8, 0},
		{AHEADS_MEDIUM_EN, 9, 0}, {AHEADS_MEDIUM, 10, 0}, {AHEADS_LARGE_V1, 11, 0},
		{AHEADS_LARGE_V2, 12, 0}, {AHEADS_LARGE_V3, 13, 0}, {AHEADS_LARGE_V3_TURBO, 14, 0},
	} {
		ctx, err := Whisper_init_with_dtw("model", test.preset, test.top)
		if err != nil || ctx == nil {
			t.Fatalf("preset %v: %v", test.preset, err)
		}
		enabled, preset, top := probeDTWState()
		if probeLegacyContextFlash() {
			t.Fatal("DTW kept incompatible flash attention")
		}
		if !enabled || preset != test.native || top != test.top {
			t.Fatalf("DTW transport %v/%d/%d", enabled, preset, top)
		}
		for token := 0; token < 2; token++ {
			stamp, err := ctx.Whisper_full_get_token_dtw_t(0, token)
			if err != nil || stamp != int64(100+token) {
				t.Fatalf("DTW output %d/%v", stamp, err)
			}
			if ctx.Whisper_full_get_token_data(0, token).TDtw() != stamp {
				t.Fatal("copied token DTW differs")
			}
		}
		for _, index := range [][2]int{{-1, 0}, {0, -1}, {1, 0}, {0, 2}} {
			if _, err := ctx.Whisper_full_get_token_dtw_t(index[0], index[1]); err == nil {
				t.Fatal("invalid DTW index accepted")
			}
		}
		for _, mode := range []int{1, 12, 20, 21} {
			probeModeSet(mode)
			if _, err := ctx.Whisper_full_get_token_dtw_t(0, 0); err == nil {
				t.Fatal("invalid native DTW result count accepted")
			}
		}
		probeModeSet(22)
		stamp, err := ctx.Whisper_full_get_token_dtw_t(0, 0)
		if err != nil || stamp != -1 || ctx.Whisper_full_get_token_data(0, 0).TDtw() != -1 {
			t.Fatal("unavailable DTW sentinel lost")
		}
		probeModeSet(0)
		if probeLiveGet() != 0 {
			t.Fatal("context params leaked")
		}
		ctx.Whisper_free()
		if probeLegacyContexts() != 0 {
			t.Fatal("context leaked")
		}
	}
	ctx := Whisper_init("model")
	if ctx == nil {
		t.Fatal("ordinary init failed")
	}
	ctx.Whisper_free()
	if enabled, _, _ := probeDTWState(); enabled {
		t.Fatal("default init enabled DTW")
	}
	if !probeLegacyContextFlash() {
		t.Fatal("ordinary init changed native flash default")
	}
	var missing *Context
	if _, err := missing.Whisper_full_get_token_dtw_t(0, 0); err == nil {
		t.Fatal("nil DTW context accepted")
	}
	calls := probeCallsGet()
	for _, test := range []struct {
		preset AlignmentHeadsPreset
		top    int
	}{
		{AHEADS_N_TOP_MOST, 0}, {AHEADS_N_TOP_MOST, -1}, {AHEADS_TINY, 1},
		{AlignmentHeadsPreset(0), 0}, {AlignmentHeadsPreset(2), 0}, {AlignmentHeadsPreset(15), 0},
	} {
		if ctx, err := Whisper_init_with_dtw("model", test.preset, test.top); err == nil || ctx != nil {
			t.Fatal("invalid DTW accepted")
		}
	}
	for _, path := range []string{"", "bad\x00path"} {
		if ctx, err := Whisper_init_with_dtw(path, AHEADS_TINY, 0); err == nil || ctx != nil {
			t.Fatal("invalid path accepted")
		}
		if Whisper_init(path) != nil {
			t.Fatal("invalid default path accepted")
		}
	}
	if probeCallsGet() != calls {
		t.Fatal("invalid options reached native initializer")
	}
	if ctx, err := Whisper_init_with_dtw("fail", AHEADS_TINY, 0); err == nil || ctx != nil {
		t.Fatal("native failure ignored")
	}
	for _, mode := range []int{10, 19} {
		probeModeSet(mode)
		if ctx, err := Whisper_init_with_dtw("model", AHEADS_TINY, 0); err == nil || ctx != nil {
			t.Fatal("native allocation/init failure ignored")
		}
		if Whisper_init("model") != nil {
			t.Fatal("default init failure ignored")
		}
		if probeLiveGet() != 0 || probeLegacyContexts() != 0 {
			t.Fatal("failure leaked context/params")
		}
	}
	probeModeSet(0)
}

func TestProbeLegacyCallbacksAcrossContexts(t *testing.T) {
	probeLegacyMode(0)
	probeLegacyExpect("concurrent", "vad")
	const workers, iterations = 8, 64
	var encoded, segments, progress atomic.Int32
	start := make(chan struct{})
	failures := make(chan error, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func(worker int) {
			defer group.Done()
			ctx := newProbeLegacyContextForWorker(worker)
			params := ctx.Whisper_full_default_params(SAMPLING_GREEDY)
			params.SetThreads(3)
			params.SetBestOf(7)
			params.SetDetectLanguage(true)
			params.SetSuppressBlank(false)
			params.SetTdrz(true)
			params.SetAltN(32)
			params.SetInitialPrompt("concurrent")
			params.SetVADModelPath("vad")
			<-start
			for iteration := 0; iteration < iterations; iteration++ {
				err := ctx.Whisper_full(params, []float32{1, 3, 5},
					func() bool {
						encoded.Add(1)
						registerEncoderBeginCallback(ctx, nil)
						runtime.Gosched()
						return true
					},
					func(n int) { segments.Add(int32(n)); registerNewSegmentCallback(ctx, nil) },
					func(n int) {
						if n != 50 {
							t.Errorf("unexpected progress %d", n)
						}
						progress.Add(1)
						registerProgressCallback(ctx, nil)
					})
				if err != nil {
					failures <- err
					return
				}
			}
		}(worker)
	}
	close(start)
	group.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	want := int32(workers * iterations)
	if encoded.Load() != want || segments.Load() != want || progress.Load() != want {
		t.Fatalf("concurrent callbacks %d/%d/%d want %d", encoded.Load(), segments.Load(), progress.Load(), want)
	}
}

func TestProbeLegacyOwnedTokenSnapshots(t *testing.T) {
	probeReset()
	probeLegacyTokensReset()
	ctx := Whisper_init("model")
	if ctx == nil {
		t.Fatal("init")
	}
	tokens, err := ctx.Whisper_token_snapshots(0)
	if err != nil || len(tokens) != 2 {
		t.Fatalf("tokens %v/%v", tokens, err)
	}
	first, second := tokens[0], tokens[1]
	if ctx.Whisper_full_get_token_id(0, 0) != first.ID {
		t.Fatal("token ID field differs from native getter")
	}

	if first.ID != 11 || first.T0 != 100 || first.T1 != 120 || first.DTW != 100 || first.P != 0.75 {
		t.Fatalf("metadata %+v", first)
	}
	if second.T0 != -1 || second.T1 != -1 || second.P != -1 || second.DTW != 101 {
		t.Fatalf("sentinels %+v", second)
	}
	if !bytes.Equal(first.Raw, []byte{0xe6, 0x97}) || first.Text != string(first.Raw) {
		t.Fatal("split bytes lost")
	}
	if len(first.Alternatives) != 2 || first.Alternatives[0].ID != 42 || first.Alternatives[0].P != 0.25 {
		t.Fatal("alternatives lost")
	}
	if !bytes.Equal(first.Alternatives[0].Raw, []byte{0xe6, 0x97}) || first.Alternatives[0].Text != string(first.Alternatives[0].Raw) {
		t.Fatal("alternative bytes lost")
	}
	if !bytes.Equal(first.Alternatives[1].Raw, bytes.Repeat([]byte{'L'}, 2048)) || first.Alternatives[1].P != -1 {
		t.Fatal("long alt/sentinel lost")
	}
	if probeLegacyAltTextCalls() <= 2 {
		t.Fatal("alternative resize unexercised")
	}
	if len(second.Alternatives) != 0 {
		t.Fatal("fabricated alts")
	}
	one, err := ctx.Whisper_token_snapshot(0, 1)
	if err != nil || !bytes.Equal(one.Raw, []byte{0xa5}) {
		t.Fatalf("one token %v/%v", one, err)
	}
	for _, index := range [][2]int{{-1, 0}, {0, -1}, {1, 0}, {0, 2}} {
		if _, err := ctx.Whisper_token_snapshot(index[0], index[1]); err == nil {
			t.Fatal("invalid token accepted")
		}
	}
	for _, segment := range []int{-1, 1} {
		if _, err := ctx.Whisper_token_snapshots(segment); err == nil {
			t.Fatal("invalid segment accepted")
		}
	}
	for _, mode := range []int{1, 12, 20, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33} {
		probeModeSet(mode)
		if _, err := ctx.Whisper_token_snapshot(0, 0); err == nil {
			t.Fatalf("token mode%d accepted", mode)
		}
		if _, err := ctx.Whisper_token_snapshots(0); err == nil {
			t.Fatalf("bulk mode%d accepted", mode)
		}
	}
	probeModeSet(21)
	empty, err := ctx.Whisper_token_snapshots(0)
	if err != nil || len(empty) != 0 {
		t.Fatal("empty token list rejected")
	}
	probeModeSet(28)
	nullText, err := ctx.Whisper_token_snapshot(0, 0)
	if err != nil || len(nullText.Raw) != 0 || len(nullText.Alternatives[0].Raw) != 0 {
		t.Fatal("null/empty text mishandled")
	}
	probeModeSet(34)
	noAlts, err := ctx.Whisper_token_snapshot(0, 0)
	if err != nil || len(noAlts.Alternatives) != 0 {
		t.Fatal("empty alternatives rejected")
	}
	probeModeSet(22)
	untimed, err := ctx.Whisper_token_snapshot(0, 0)
	if err != nil || untimed.DTW != -1 {
		t.Fatal("unavailable DTW lost")
	}
	probeModeSet(0)
	var missing *Context
	if _, err := missing.Whisper_token_snapshot(0, 0); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := missing.Whisper_token_snapshots(0); err == nil {
		t.Fatal("nil bulk context accepted")
	}
	probeLegacyTokensMutate()
	ctx.Whisper_free()
	if !bytes.Equal(first.Raw, []byte{0xe6, 0x97}) || !bytes.Equal(first.Alternatives[0].Raw, []byte{0xe6, 0x97}) || !bytes.Equal(first.Alternatives[1].Raw, bytes.Repeat([]byte{'L'}, 2048)) {
		t.Fatal("snapshot borrowed native storage")
	}
	if string(append(append([]byte{}, first.Raw...), second.Raw...)) != "日" {
		t.Fatal("split tokens did not reconstruct text")
	}
	probeLegacyTokensReset()
}
