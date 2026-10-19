//go:build probe

package whisper

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestProbeHighLevelParallelCapabilityProgressAndCleanup(t *testing.T) {
	probeParallelReset()
	path := filepath.Join(t.TempDir(), "model")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx, err := m.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	ctx.SetThreads(3)
	ctx.SetBestOf(7)
	ctx.SetDetectLanguage(true)
	ctx.SetSuppressBlank(false)
	ctx.SetTdrz(true)
	ctx.SetAltN(32)
	ctx.SetInitialPrompt("")
	ctx.SetVADModelPath("")
	implementation := ctx.(*context)
	before := implementation.params.String()
	pcm := []float32{1, 3, 5}
	if _, ok := ctx.(ParallelProcessor); !ok {
		t.Fatal("optional capability missing")
	}
	for _, processors := range []int{0, -1, int(1 << 31)} {
		calls := probeParallelCalls()
		if err := ProcessParallel(ctx, pcm, processors, nil, nil, nil); err == nil {
			t.Fatal("invalid processor count reached live context")
		}
		if err := ctx.(ParallelProcessor).ProcessParallel(pcm, processors, nil, nil, nil); err == nil {
			t.Fatal("direct optional method accepted invalid processor count")
		}
		if probeParallelCalls() != calls {
			t.Fatal("invalid count reached native pointer route")
		}
	}
	calls := probeParallelCalls()
	if err := ProcessParallel(ctx, nil, 2, nil, nil, nil); err == nil {
		t.Fatal("empty PCM accepted")
	}
	if probeParallelCalls() != calls {
		t.Fatal("empty PCM reached native pointer route")
	}
	for _, processors := range []int{2, 1, 2} {
		encoded, segments, progress := 0, 0, 0
		calls := probeParallelCalls()
		err := ProcessParallel(ctx, pcm, processors,
			func() bool { encoded++; return true },
			func(s Segment) {
				segments++
				if s.Text != "日" || s.Num != 0 || len(s.Tokens) != 2 {
					t.Errorf("segment metadata: %+v", s)
				}
				if len(s.Tokens) == 2 && (s.Tokens[0].Id != 11 || s.Tokens[0].P != .75 ||
					s.Tokens[0].Start != time.Second || s.Tokens[1].Start != -10*time.Millisecond) {
					t.Errorf("token metadata: %+v", s.Tokens)
				}
			},
			func(p int) {
				progress++
				if p != 50 {
					t.Errorf("progress %d", p)
				}
			})
		if err != nil {
			t.Fatal(err)
		}
		if encoded != 1 || segments != 1 || progress != 1 {
			t.Fatalf("callback counts %d/%d/%d", encoded, segments, progress)
		}
		expected := calls
		if processors > 1 {
			expected++
		}
		if probeParallelCalls() != expected {
			t.Fatal("wrong native dispatch route")
		}
		if implementation.params.String() != before {
			t.Fatal("per-call single_segment setting mutated context defaults")
		}
		if _, err := ctx.NextSegment(); err != nil {
			t.Fatal(err)
		}
		if _, err := ctx.NextSegment(); err != io.EOF {
			t.Fatalf("cursor not exhausted: %v", err)
		}
		if probeParallelLive() != 0 {
			t.Fatal("parameter/string leak")
		}
	}
	// The existing sequential method shares the local-options/iterator path.
	if err := ctx.Process(pcm, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := ctx.NextSegment(); err != nil {
		t.Fatal(err)
	}
	probeParallelFailure(1)
	if err := ProcessParallel(ctx, pcm, 2, nil, nil, nil); err == nil {
		t.Fatal("native failure ignored")
	}
	probeParallelFailure(0)
	if err := ProcessParallel(ctx, pcm, 2, func() bool { return false }, nil, nil); err == nil {
		t.Fatal("encoder abort ignored")
	}
	if err := ProcessParallel(ctx, pcm, 2, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if probeParallelLive() != 0 {
		t.Fatal("failure cleanup leaked allocations")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	calls = probeParallelCalls()
	if err := ProcessParallel(ctx, pcm, 2, nil, nil, nil); err == nil {
		t.Fatal("closed model accepted")
	}
	if probeParallelCalls() != calls {
		t.Fatal("closed model reached native decoder")
	}
}

func TestProbeHighLevelParallelRejectsUnsupportedAndInvalidInput(t *testing.T) {
	probeParallelReset()
	calls := probeParallelNativeCalls()
	var typedNil *context
	var unsupported Context = struct{ Context }{nil}
	for _, ctx := range []Context{nil, typedNil, unsupported} {
		if err := ProcessParallel(ctx, []float32{1, 3, 5}, 2, nil, nil, nil); err == nil {
			t.Fatal("nil/unsupported context accepted")
		}
	}
	for _, processors := range []int{0, -1, int(1 << 31)} {
		if err := ProcessParallel(typedNil, []float32{1, 3, 5}, processors, nil, nil, nil); err == nil {
			t.Fatal("invalid processors accepted")
		}
	}
	if probeParallelNativeCalls() != calls {
		t.Fatal("rejected input reached native decoder")
	}
}

func TestProbeHighLevelCallbackCloseIsRejected(t *testing.T) {
	probeParallelReset()
	path := filepath.Join(t.TempDir(), "model")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx, err := m.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	ctx.SetThreads(3)
	ctx.SetBestOf(7)
	ctx.SetDetectLanguage(true)
	ctx.SetSuppressBlank(false)
	ctx.SetTdrz(true)
	ctx.SetAltN(32)
	ctx.SetInitialPrompt("")
	ctx.SetVADModelPath("")
	called := false
	err = ctx.Process([]float32{1, 3, 5}, func() bool {
		called = true
		if closeErr := m.Close(); closeErr == nil || !strings.Contains(closeErr.Error(), "busy") {
			t.Errorf("callback close must reject live inference, got %v", closeErr)
		}
		return false
	}, nil, nil)
	if !called || err == nil {
		t.Fatal("encoder abort not observed")
	}
	if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); err != nil {
		t.Fatal("context not usable after callback rejection:", err)
	}
}
