// Concrete stream ABI for wrapper lifetime probes; aborting unrelated stubs
// must never stand in for an exercised operation.
#include "crispasr_session.h"
#include <string.h>
static int stream_closes, stream_live;
static crispasr_session* stream_owner;
int probe_streams_for_owner(crispasr_session* owner) {
    return owner == stream_owner ? stream_live : 0;
}
int probe_stream_closes(void) {
    return stream_closes;
}
crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int threads, int step, int length, int keep,
                                              const char* lang, int translate) {
    ++stream_live;
    stream_owner = s;
    return (crispasr_stream*)2;
}
#ifndef CRISPASR_PROBE_EXISTING_STREAM_OPS
void crispasr_stream_close(crispasr_stream* s) {
    ++stream_closes;
    --stream_live;
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

int crispasr_session_stream_kind(crispasr_session* session) {
    return 1;
}
