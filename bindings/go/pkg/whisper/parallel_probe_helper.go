//go:build probe

package whisper

/*
void probe_reset(void);
void probe_legacy_mode(int);
void probe_legacy_expect(const char*, const char*);
int probe_parallel_by_ref_calls(void);
void probe_legacy_foreign_callback(int);
int probe_rust_legacy_contexts(void);
int probe_legacy_calls(void);
int probe_live_get(void);
static void probe_parallel_reset(void) { probe_reset(); probe_legacy_mode(0); probe_legacy_expect("", ""); }
*/
import "C"

func probeParallelReset()           { C.probe_parallel_reset() }
func probeParallelCalls() int       { return int(C.probe_parallel_by_ref_calls()) }
func probeParallelNativeCalls() int { return int(C.probe_legacy_calls()) }
func probeParallelLive() int        { return int(C.probe_live_get()) }
func probeParallelFailure(mode int) { C.probe_legacy_mode(C.int(mode)) }

func probeParallelForeignCallback(enabled bool) {
	if enabled {
		C.probe_legacy_foreign_callback(1)
	} else {
		C.probe_legacy_foreign_callback(0)
	}
}
func probeParallelContexts() int { return int(C.probe_rust_legacy_contexts()) }
