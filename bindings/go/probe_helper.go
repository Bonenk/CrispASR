//go:build probe

package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../ggml/include
#include <stdlib.h>
#include "crispasr_session.h"
#include "crispasr.h"
#include "crispasr_chat.h"
typedef struct crispasr_session CrispasrSession;
void probe_mode_set(int m);
void probe_db_mode(int mode);
int probe_db_live(void);
int probe_db_calls(void);
void probe_titanet_mode(int mode);
int probe_titanet_live(void);
int probe_live_get(void);
int probe_calls_get(void);
int probe_result_frees_get(void);
void probe_reset(void);
void probe_mutate(void);
void probe_utility_mode(int mode);
int probe_utility_live(void);
int probe_stream_closes_get(void);
*/
import "C"
import (
	"testing"
	"unsafe"
)

func probeModeSet(m int)       { C.probe_mode_set(C.int(m)) }
func probeLiveGet() int        { return int(C.probe_live_get()) }
func probeCallsGet() int       { return int(C.probe_calls_get()) }
func probeResultFreesGet() int { return int(C.probe_result_frees_get()) }
func probeReset()              { C.probe_reset() }
func probeMutate()             { C.probe_mutate() }

func newProbeSession(t *testing.T) *CrispasrSession {
	t.Helper()
	h := (*C.CrispasrSession)(unsafe.Pointer(uintptr(0x100000001)))
	return &CrispasrSession{handle: h}
}

func probeUtilityMode(mode int) { C.probe_utility_mode(C.int(mode)) }
func probeUtilityLive() int     { return int(C.probe_utility_live()) }
func probeStreamCloses() int    { return int(C.probe_stream_closes_get()) }

func probeTitaNetMode(mode int) { C.probe_titanet_mode(C.int(mode)) }
func probeTitaNetLive() int     { return int(C.probe_titanet_live()) }

func probeDbMode(mode int) { C.probe_db_mode(C.int(mode)) }
func probeDbLive() int     { return int(C.probe_db_live()) }
func probeDbCalls() int    { return int(C.probe_db_calls()) }
