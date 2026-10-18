// Model-free chat ABI transport, including bool/size_t layout and callbacks.
#include "crispasr_chat.h"
#include <stdlib.h>
#include <string.h>
struct crispasr_chat_session {
    crispasr_chat_abort_callback abort;
    void* user;
    int context;
};
static int live_sessions, live_strings, reset_calls;
int probe_chat_sessions(void) {
    return live_sessions;
}
int probe_chat_strings(void) {
    return live_strings;
}
int probe_chat_resets(void) {
    return reset_calls;
}
int probe_chat_open_size(void) {
    return sizeof(crispasr_chat_open_params);
}
int probe_chat_generate_size(void) {
    return sizeof(crispasr_chat_generate_params);
}
void crispasr_chat_open_params_default(crispasr_chat_open_params* out) {
    memset(out, 0, sizeof(*out));
    out->n_threads = 4;
    out->n_ctx = 1024;
    out->n_gpu_layers = -1;
    out->use_mmap = true;
}
void crispasr_chat_generate_params_default(crispasr_chat_generate_params* out) {
    memset(out, 0, sizeof(*out));
    out->max_tokens = 256;
    out->temperature = .8f;
    out->top_k = 40;
    out->top_p = .95f;
    out->repeat_penalty = 1.1f;
}
static void error(crispasr_chat_error* e, int code, const char* message) {
    if (e) {
        e->code = code;
        strncpy(e->message, message, 255);
        e->message[255] = 0;
    }
}
crispasr_chat_session_t crispasr_chat_open(const char* path, const crispasr_chat_open_params* p,
                                           crispasr_chat_error* e) {
    if (!strcmp(path, "fail")) {
        error(e, 3, "cannot open");
        return NULL;
    }
    if (p && p->embeddings) {
        error(e, 4, "embeddings must be false");
        return NULL;
    }
    if (p && p->chat_template && (strcmp(p->chat_template, "gemma") || p->use_mmap || p->n_gpu_layers != 0)) {
        error(e, 5, "layout/options mismatch");
        return NULL;
    }
    struct crispasr_chat_session* s = calloc(1, sizeof(*s));
    s->context = p ? p->n_ctx : 1024;
    ++live_sessions;
    return s;
}
void crispasr_chat_close(crispasr_chat_session_t s) {
    if (s) {
        free(s);
        --live_sessions;
    }
}
int32_t crispasr_chat_reset(crispasr_chat_session_t s, crispasr_chat_error* e) {
    ++reset_calls;
    return 0;
}
static int ready(crispasr_chat_session_t s, crispasr_chat_error* e) {
    if (s->abort && !s->abort(s->user)) {
        error(e, 40, "cancelled");
        return 0;
    }
    return 1;
}
char* crispasr_chat_generate(crispasr_chat_session_t s, const crispasr_chat_message* m, size_t n,
                             const crispasr_chat_generate_params* p, crispasr_chat_error* e) {
    if (!ready(s, e))
        return NULL;
    if (n != 2 || strcmp(m[0].role, "system") || strcmp(m[1].content, "日本語")) {
        error(e, 6, "message transport mismatch");
        return NULL;
    }
    if (p && p->n_stop &&
        (p->n_stop != 1 || strcmp(p->stop[0], "STOP") || p->max_tokens != 7 || p->seed != 0xffffffffu)) {
        error(e, 7, "generate layout mismatch");
        return NULL;
    }
    const char* reply = p && p->prefill_only ? "" : "Grüße";
    char* out = malloc(strlen(reply) + 1);
    strcpy(out, reply);
    ++live_strings;
    return out;
}
int32_t crispasr_chat_generate_stream(crispasr_chat_session_t s, const crispasr_chat_message* m, size_t n,
                                      const crispasr_chat_generate_params* p, crispasr_chat_on_token cb, void* user,
                                      crispasr_chat_error* e) {
    if (!ready(s, e))
        return 40;
    char head[] = {(char)0xe6, (char)0x97, 0}, tail[] = {(char)0xa5, 0};
    cb(head, user);
    head[0] = 'X';
    if (!ready(s, e))
        return 40;
    cb(tail, user);
    tail[0] = 'X';
    return 0;
}
void crispasr_chat_set_abort_callback(crispasr_chat_session_t s, crispasr_chat_abort_callback cb, void* user) {
    s->abort = cb;
    s->user = user;
}
const char* crispasr_chat_template_name(crispasr_chat_session_t s) {
    return "probe-template";
}
int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s) {
    return s->context;
}
int32_t crispasr_chat_count_tokens(crispasr_chat_session_t s, const crispasr_chat_message* m, size_t n,
                                   crispasr_chat_error* e) {
    return n ? strlen(m[n - 1].content) : 0;
}
size_t crispasr_chat_memory_estimate(const char* path, const crispasr_chat_open_params* p, crispasr_chat_error* e) {
#if SIZE_MAX > UINT32_MAX
    return ((size_t)1 << 40) + 123;
#else
    return 123;
#endif
}
void crispasr_chat_string_free(char* text) {
    if (text) {
        free(text);
        --live_strings;
    }
}
const char* crispasr_chat_ai_disclosure_text(void) {
    return "You are talking to an AI.";
}
