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

func TestProbeModelTokenSnapshotsWithoutChangingContextInterface(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	model, err := NewWithDTW(path, native.AHEADS_TINY_EN, 0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := model.NewContext()
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := TokenSnapshots(ctx, 0)
	if err != nil || len(tokens) != 2 {
		t.Fatalf("snapshots %v/%v", tokens, err)
	}
	if tokens[0].DTW != 100 || tokens[0].ID != 11 || string(tokens[0].Raw) != string([]byte{0xe6, 0x97}) || tokens[0].Alternatives[0].ID != 42 {
		t.Fatal("high-level details dropped")
	}
	if _, ok := ctx.(TokenSnapshotProvider); !ok {
		t.Fatal("optional capability missing")
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := TokenSnapshots(ctx, 0); err == nil {
		t.Fatal("closed model accepted")
	}
	if _, err := TokenSnapshots(nil, 0); err == nil {
		t.Fatal("nil context accepted")
	}
	var typedNil *context
	if _, err := TokenSnapshots(typedNil, 0); err == nil {
		t.Fatal("typed nil context accepted")
	}
	// Embedding the old interface keeps old implementations source-compatible.
	var old Context = struct{ Context }{nil}
	if _, err := TokenSnapshots(old, 0); err == nil {
		t.Fatal("unsupported capability accepted")
	}
	if tokens[0].DTW != 100 || tokens[0].ID != 11 {
		t.Fatal("snapshot tied to closed model")
	}
}
