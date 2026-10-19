//go:build probe

package whisper

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func newLifecycleProbe(t *testing.T) (Model, Context) {
	t.Helper()
	probeParallelReset()
	path := filepath.Join(t.TempDir(), "model")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error("cleanup:", err)
		}
	})
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
	return m, ctx
}
func expectLifecyclePanic(t *testing.T, want error, call func()) {
	t.Helper()
	defer func() {
		value := recover()
		err, ok := value.(error)
		if !ok || !errors.Is(err, want) {
			t.Errorf("panic = %v, want %v", value, want)
		}
	}()
	call()
	t.Error("native operation did not reject invalid lifecycle")
}
func TestProbeHighLevelConcurrentLifetimeAndOptions(t *testing.T) {
	m, ctx := newLifecycleProbe(t)
	other, err := m.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- ctx.Process([]float32{1, 3, 5}, func() bool { close(entered); <-release; return true }, nil, nil)
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("native callback never entered")
	}
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	nativeCalls := probeParallelNativeCalls()
	for _, target := range []Context{ctx, other} {
		if err := target.Process([]float32{1, 3, 5}, nil, nil, nil); !errors.Is(err, ErrModelBusy) {
			t.Errorf("overlap: %v", err)
		}
		if err := ProcessParallel(target, []float32{1, 3, 5}, 2, nil, nil, nil); !errors.Is(err, ErrModelBusy) {
			t.Errorf("parallel overlap: %v", err)
		}
		if _, err := target.NextSegment(); !errors.Is(err, ErrModelBusy) {
			t.Errorf("cursor overlap: %v", err)
		}
		if _, err := TokenSnapshots(target, 0); !errors.Is(err, ErrModelBusy) {
			t.Errorf("token overlap: %v", err)
		}
		if err := target.SetLanguage("auto"); !errors.Is(err, ErrModelBusy) {
			t.Errorf("native setter overlap: %v", err)
		}
	}
	if _, err := m.NewContext(); !errors.Is(err, ErrModelBusy) {
		t.Errorf("new context overlap: %v", err)
	}
	if err := m.Close(); !errors.Is(err, ErrModelBusy) {
		t.Errorf("close overlap: %v", err)
	}
	for _, fn := range []func(){func() { m.IsMultilingual() }, func() { m.Languages() }, func() { ctx.IsMultilingual() },
		func() { ctx.DetectedLanguage() }, func() { ctx.ResetTimings() }, func() { ctx.PrintTimings() },
		func() { ctx.IsBEG(Token{}) }, func() { ctx.IsSOT(Token{}) }, func() { ctx.IsEOT(Token{}) },
		func() { ctx.IsPREV(Token{}) }, func() { ctx.IsSOLM(Token{}) }, func() { ctx.IsNOT(Token{}) },
		func() { ctx.IsLANG(Token{}, "en") }, func() { ctx.IsText(Token{}) }} {
		expectLifecyclePanic(t, ErrModelBusy, fn)
	}
	if _, err := ctx.(*context).WhisperLangAutoDetect(0, 3); !errors.Is(err, ErrModelBusy) {
		t.Errorf("LID overlap: %v", err)
	}
	if !strings.Contains(fmt.Sprint(m), "busy") {
		t.Error("unsafe String during native call")
	}
	if probeParallelNativeCalls() != nativeCalls || probeParallelContexts() != 1 {
		t.Error("rejected calls entered/freed native context")
	}
	// Per-context Go-owned option setters remain usable from callbacks; their
	// updates affect the next snapshot, not the already-running C parameter copy.
	ctx.SetThreads(5)
	ctx.SetInitialPrompt("next call")
	if !strings.Contains(ctx.SystemInfo(), "n_threads = 5") {
		t.Error("option update lost")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal("active decode changed by option update:", err)
	}
	ctx.SetThreads(3)
	ctx.SetInitialPrompt("")
	if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); err != nil {
		t.Fatal("borrow was not released:", err)
	}
	saved, err := ctx.NextSegment()
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if probeParallelContexts() != 0 {
		t.Fatal("native context double-close/leak")
	}
	if saved.Text != "日" || len(saved.Tokens) != 2 {
		t.Fatal("owned result changed after close")
	}
	if _, err := other.NextSegment(); !errors.Is(err, ErrInternalAppError) {
		t.Error("other context survived closed model")
	}
	if _, err := TokenSnapshots(ctx, 0); !errors.Is(err, ErrInternalAppError) {
		t.Error("closed model token copy reached native")
	}
	expectLifecyclePanic(t, ErrInternalAppError, func() { ctx.IsText(Token{}) })
}
func TestProbeHighLevelForeignCallbackCloseAndReentry(t *testing.T) {
	m, ctx := newLifecycleProbe(t)
	probeParallelForeignCallback(true)
	defer probeParallelForeignCallback(false)
	for _, processors := range []int{1, 2} {
		callbacks := 0
		err := ProcessParallel(ctx, []float32{1, 3, 5}, processors, func() bool {
			callbacks++
			if err := m.Close(); !errors.Is(err, ErrModelBusy) {
				t.Errorf("foreign close: %v", err)
			}
			if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); !errors.Is(err, ErrModelBusy) {
				t.Errorf("foreign reentry: %v", err)
			}
			ctx.SetThreads(3)
			return false
		}, nil, nil)
		if callbacks != 1 || err == nil {
			t.Fatal("foreign callback/abort not exercised")
		}
		if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); err != nil {
			t.Fatal("foreign callback leaked borrow:", err)
		}
	}
}
func TestProbeHighLevelConcurrentOptionSnapshots(t *testing.T) {
	_, ctx := newLifecycleProbe(t)
	language := ctx.Language()
	var writes sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 2; i++ {
		writes.Add(1)
		go func() {
			defer writes.Done()
			<-start
			for j := 0; j < 100; j++ {
				ctx.SetThreads(3)
				ctx.SetInitialPrompt("")
				ctx.SetVADModelPath("")
				if ctx.Language() != language {
					t.Error("language option corrupted")
				}
			}
		}()
	}
	close(start)
	for i := 0; i < 100; i++ {
		if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); err != nil {
			t.Error(err)
		}
	}
	writes.Wait()
	if probeParallelLive() != 0 {
		t.Error("temporary parameter storage leaked")
	}
}
func TestProbeHighLevelIdleNativeQueriesAndLastLanguage(t *testing.T) {
	m, ctx := newLifecycleProbe(t)
	if !m.IsMultilingual() || !ctx.IsMultilingual() {
		t.Error("multilingual adapter")
	}
	if !reflect.DeepEqual(m.Languages(), []string{"en", "de"}) {
		t.Error("last supported language omitted")
	}
	if err := ctx.SetLanguage("de"); err != nil {
		t.Fatal(err)
	}
	if ctx.Language() != "de" || ctx.DetectedLanguage() != "en" {
		t.Error("language adapter")
	}
	if err := ctx.SetLanguage("bad"); !errors.Is(err, ErrUnsupportedLanguage) {
		t.Error("invalid language accepted")
	}
	probs, err := ctx.(*context).WhisperLangAutoDetect(0, 3)
	if err != nil || !reflect.DeepEqual(probs, []float32{.75, .25}) {
		t.Errorf("LID: %v %v", probs, err)
	}
	ctx.PrintTimings()
	ctx.ResetTimings()
	if !ctx.IsText(Token{Id: 11}) || ctx.IsText(Token{Id: 100}) || !ctx.IsBEG(Token{Id: 105}) ||
		!ctx.IsSOT(Token{Id: 101}) || !ctx.IsEOT(Token{Id: 100}) || !ctx.IsPREV(Token{Id: 102}) ||
		!ctx.IsSOLM(Token{Id: 103}) || !ctx.IsNOT(Token{Id: 104}) || !ctx.IsLANG(Token{Id: 201}, "de") || ctx.IsLANG(Token{}, "bad") {
		t.Error("token classification changed")
	}
}

func TestProbeHighLevelConcurrentCloseFreesOnce(t *testing.T) {
	m, ctx := newLifecycleProbe(t)
	start := make(chan struct{})
	results := make(chan error, 16)
	var calls sync.WaitGroup
	for i := 0; i < 16; i++ {
		calls.Add(1)
		go func() { defer calls.Done(); <-start; results <- m.Close() }()
	}
	close(start)
	calls.Wait()
	close(results)
	for err := range results {
		if err != nil && !errors.Is(err, ErrModelBusy) {
			t.Error(err)
		}
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if probeParallelContexts() != 0 {
		t.Fatal("concurrent close freed model twice or leaked")
	}
	if err := ctx.Process([]float32{1, 3, 5}, nil, nil, nil); !errors.Is(err, ErrInternalAppError) {
		t.Error("closed decode:", err)
	}
}
