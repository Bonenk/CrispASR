// Hosted real-model acceptance for the public high-level Go lifecycle.
package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	whisper "github.com/CrispStrobe/CrispASR/bindings/go/pkg/whisper"
)

func require(ok bool, message string) {
	if !ok {
		panic(message)
	}
}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
func pcm(path string) []float32 {
	data, err := os.ReadFile(path)
	check(err)
	require(string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE", "Not RIFF WAV")
	format, channels, rate, bits := uint16(0), uint16(0), uint32(0), uint16(0)
	var samples []float32
	for i := 12; i+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
		begin := i + 8
		require(size >= 0 && begin+size <= len(data), "Truncated WAV")
		switch string(data[i : i+4]) {
		case "fmt ":
			require(size >= 16, "Short WAV format")
			format = binary.LittleEndian.Uint16(data[begin:])
			channels = binary.LittleEndian.Uint16(data[begin+2:])
			rate = binary.LittleEndian.Uint32(data[begin+4:])
			bits = binary.LittleEndian.Uint16(data[begin+14:])
		case "data":
			require(size%2 == 0, "Unaligned PCM")
			for j := begin; j < begin+size; j += 2 {
				samples = append(samples, float32(int16(binary.LittleEndian.Uint16(data[j:])))/32768)
			}
		}
		i = begin + size + (size % 2)
	}
	require(format == 1 && channels == 1 && rate == 16000 && bits == 16 && len(samples) > 16000, "Unexpected fixture format")
	return samples
}

type outcome struct {
	Name           string
	Segments       []whisper.Segment
	CallbackChecks int
}

func main() {
	require(len(os.Args) == 4, "model, WAV and output paths required")
	audio := pcm(os.Args[2])
	m, err := whisper.New(os.Args[1])
	check(err)
	defer m.Close()
	ctx, err := m.NewContext()
	check(err)
	other, err := m.NewContext()
	check(err)
	ctx.SetThreads(2)
	ctx.SetBestOf(1)
	ctx.SetTemperature(0)
	ctx.SetTemperatureFallback(0)
	ctx.SetTokenTimestamps(true)
	ctx.SetAltN(2)
	results := []outcome{}
	run := func(label string, guard bool, processors int) {
		var checks atomic.Int32
		encoder := whisper.EncoderBeginCallback(nil)
		if guard {
			encoder = func() bool {
				require(errors.Is(m.Close(), whisper.ErrModelBusy), "Callback close not rejected")
				checks.Add(1)
				require(errors.Is(other.Process(audio, nil, nil, nil), whisper.ErrModelBusy), "Callback reentry not rejected")
				checks.Add(1)
				_, e := m.NewContext()
				require(errors.Is(e, whisper.ErrModelBusy), "Callback context construction not rejected")
				checks.Add(1)
				_, e = ctx.NextSegment()
				require(errors.Is(e, whisper.ErrModelBusy), "Callback cursor read not rejected")
				checks.Add(1)
				ctx.SetInitialPrompt("") // synchronized option update applies to next call
				return true
			}
		}
		check(whisper.ProcessParallel(ctx, audio, processors, encoder, nil, nil))
		segments := []whisper.Segment{}
		for {
			segment, e := ctx.NextSegment()
			if e == io.EOF {
				break
			}
			check(e)
			require(segment.Start >= 0 && segment.End > segment.Start && segment.End <= time.Duration(len(audio))*time.Second/16000, "Invalid segment bounds")
			for _, token := range segment.Tokens {
				require(!math.IsNaN(float64(token.P)), "Nonfinite probability")
			}
			segments = append(segments, segment)
		}
		require(len(segments) > 0, "No model output")
		if guard {
			require(checks.Load() >= 4, "No actual native encoder callback")
		}
		text := ""
		for _, segment := range segments {
			text += segment.Text
		}
		require(strings.Contains(strings.ToLower(text), "country"), "Unexpected JFK transcription")
		snapshots, e := whisper.TokenSnapshots(ctx, 0)
		check(e)
		require(len(snapshots) > 0, "Missing owned token metadata")
		encoded, e := json.Marshal(snapshots)
		check(e)
		results = append(results, outcome{label, segments, int(checks.Load())})
		partial, e := json.Marshal(results)
		check(e)
		check(os.WriteFile(os.Args[3], partial, 0600))
		// Retain exact metadata independently of the next native decoder mutation.
		check(os.WriteFile(os.Args[3]+"-"+label+"-tokens.json", encoded, 0600))
	}
	run("baseline", false, 1)
	run("callback-guard", true, 1)
	require(reflect.DeepEqual(results[0].Segments, results[1].Segments), "Callback guard changed decoded output")
	// Hold a genuine encoder callback while another goroutine tries native access.
	entered, release := make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	go func() { done <- ctx.Process(audio, func() bool { close(entered); <-release; return false }, nil, nil) }()
	select {
	case <-entered:
	case <-time.After(30 * time.Second):
		panic("Encoder callback timeout")
	}
	require(errors.Is(m.Close(), whisper.ErrModelBusy), "Concurrent close not rejected")
	require(errors.Is(other.Process(audio, nil, nil, nil), whisper.ErrModelBusy), "Concurrent decode not rejected")
	close(release)
	select {
	case e := <-done:
		// Native encoder-begin false ends decode with status zero.
		_ = e
		_, cursorError := ctx.NextSegment()
		require(errors.Is(cursorError, io.EOF), "Aborted decode retained output")
	case <-time.After(30 * time.Second):
		panic("Aborted decode timeout")
	}
	run("after-abort", false, 1)
	require(reflect.DeepEqual(results[0].Segments, results[2].Segments), "Recovery changed decoded output")
	run("parallel", true, 2)
	saved, err := json.Marshal(results)
	check(err)
	var wait sync.WaitGroup
	for i := 0; i < 8; i++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			e := m.Close()
			if e != nil && !errors.Is(e, whisper.ErrModelBusy) {
				panic(e)
			}
		}()
	}
	wait.Wait()
	check(m.Close())
	_, err = m.NewContext()
	require(err != nil, "Closed model accepted context")
	after, err := json.Marshal(results)
	check(err)
	require(string(saved) == string(after), "Owned results changed after close")
	check(os.WriteFile(os.Args[3], saved, 0600))
	fmt.Println("GO_REAL_MODEL_LIFECYCLE_PASS: 4 decodes, callback and concurrent close/reentry, abort recovery, parallel, owned results")
}
