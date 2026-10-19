//go:build probe

package whisper

import (
	"math"
	"reflect"
	"testing"
)

func TestProbeGenericSpeakerOwnershipAndReturnCodes(t *testing.T) {
	probeSpeakerMode(0)
	embedder, err := NewSpeakerEmbedder("embed", 2, "cache")
	if err != nil {
		t.Fatal(err)
	}
	defer embedder.Close()
	if embedder.Dim() != 3 || embedder.Name() != "generic" || !reflect.DeepEqual(embedder.Embed([]float32{1, 3, 5}), []float32{.25, .5, .75}) {
		t.Fatal("actual generic embed/name/dim transport")
	}
	for _, mode := range []int{4, 8} {
		probeSpeakerMode(mode)
		if out := embedder.Embed([]float32{1, 3, 5}); out != nil {
			t.Error("accepted malformed success code", mode, out)
		}
	}
	probeSpeakerMode(2)
	if out := embedder.Embed([]float32{1, 3, 5}); out != nil {
		t.Error("accepted native failure")
	}
	probeSpeakerMode(3)
	if out := embedder.Embed([]float32{1, 3, 5}); out != nil {
		t.Error("invalid dimension")
	}
	probeSpeakerMode(0)
	embedder.Close()
	embedder.Close()
	if embedder.Embed([]float32{1, 3, 5}) != nil || embedder.Dim() != 0 || embedder.Name() != "" || probeSpeakerLive() != 0 {
		t.Error("closed resource")
	}
}
func TestProbeGenericClusterBoundsAndNativeOutput(t *testing.T) {
	probeSpeakerMode(0)
	// No allocation or native call is needed to reproduce the old n*dim overflow.
	if _, err := AgglomerativeCluster(nil, 2, int(^uint(0)>>1), .6, 3); err == nil {
		t.Error("oversized dimension accepted")
	}
	for _, threshold := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := AgglomerativeCluster([]float32{1, 2, 3, 4, 5, 6}, 2, 3, threshold, 3); err == nil {
			t.Error("nonfinite threshold")
		}
	}
	for _, mode := range []int{0, 2, 6, 7, 9} {
		probeSpeakerMode(mode)
		out, err := AgglomerativeCluster([]float32{1, 2, 3, 4, 5, 6}, 2, 3, .6, 3)
		if mode == 0 {
			if err != nil || !reflect.DeepEqual(out, []int32{0, 1}) {
				t.Error(out, err)
			}
		} else if mode == 9 {
			if err != nil || !reflect.DeepEqual(out, []int32{-1, -1}) {
				t.Error("unassigned cluster result", out, err)
			}
		} else if err == nil {
			t.Error("invalid native cluster count accepted", mode, out)
		}
	}
	probeSpeakerMode(0)
}
func TestProbeGenericPyannoteOwnedResource(t *testing.T) {
	probeSpeakerMode(0)
	cache, err := NewPyannoteCache([]float32{1, 3, 5}, "seg", 2)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()
	segs := []DiarizeSeg{{T0: 125, T1: 175, Speaker: -1}, {T0: 200, T1: 250, Speaker: -1}}
	if err := cache.Apply(segs, 125); err != nil || segs[0].Speaker != 2 || segs[1].Speaker != 1 {
		t.Error("cache transport", err, segs)
	}
	for _, mode := range []int{2, 5} {
		probeSpeakerMode(mode)
		if err := cache.Apply(segs, 125); err == nil {
			t.Error("native cache failure accepted", mode)
		}
	}
	probeSpeakerMode(0)
	cache.Close()
	cache.Close()
	if err := cache.Apply(segs, 125); err == nil || probeSpeakerLive() != 0 {
		t.Error("cache close")
	}
}

func TestProbeGenericSpeakerRejectsNulAndOversizedOptions(t *testing.T) {
	probeSpeakerMode(0)
	for _, pair := range [][2]string{{"embed\x00bad", "cache"}, {"embed", "cache\x00bad"}} {
		e, err := NewSpeakerEmbedder(pair[0], 2, pair[1])
		if err == nil {
			e.Close()
			t.Error("NUL string accepted")
		}
	}
	c, err := NewPyannoteCache([]float32{1, 3, 5}, "seg\x00bad", 2)
	if err == nil {
		c.Close()
		t.Error("NUL cache path accepted")
	}
	if probeSpeakerLive() != 0 {
		t.Error("constructor failure leak")
	}
}
