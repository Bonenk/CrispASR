// Concrete stream ABI for wrapper lifetime probes; aborting unrelated stubs
// must never stand in for an exercised operation.
#include "crispasr_session.h"
#include <string.h>
static int stream_closes;
int probe_stream_closes(void) {
    return stream_closes;
}
crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int threads, int step, int length, int keep,
                                              const char* lang, int translate) {
    return (crispasr_stream*)2;
}
#ifndef CRISPASR_PROBE_EXISTING_STREAM_OPS
void crispasr_stream_close(crispasr_stream* s) {
    ++stream_closes;
}
int crispasr_stream_feed(crispasr_stream* s, const float* p, int n) {
    return 1;
}
int crispasr_stream_flush(crispasr_stream* s) {
    return 0;
}
#endif
void crispasr_stream_set_live_decode(crispasr_stream* s, int enabled) {}
int crispasr_stream_get_text(crispasr_stream* s, char* text, int cap, double* t0, double* t1, int64_t* count) {
    if (cap < 3)
        return -1;
    memcpy(text, "ok", 3);
    *t0 = 1;
    *t1 = 2;
    *count = 3;
    return 2;
}
