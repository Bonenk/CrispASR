//go:build probe

package whisper

import (
	"os"
	"path/filepath"
	"testing"

	native "github.com/CrispStrobe/CrispASR/bindings/go"
)

func TestProbeModelDTWConstructor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	model, err := NewWithDTW(path, native.AHEADS_TINY_EN, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	model, err = New(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := NewWithDTW(path, native.AHEADS_N_TOP_MOST, 0); err == nil {
		t.Fatal("invalid top count accepted")
	}
	if _, err := NewWithDTW(path+".missing", native.AHEADS_TINY_EN, 0); err == nil {
		t.Fatal("missing model accepted")
	}
}
