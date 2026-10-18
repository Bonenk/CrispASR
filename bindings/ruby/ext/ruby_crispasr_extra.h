/* Included by ruby_crispasr_session.c after its legacy declarations. */
#include <limits.h>
extern void crispasr_pcm_free(float* pcm);
extern int crispasr_session_beats(struct CrispasrSession* s, const float* pcm, int n_samples, int sample_rate);
extern const float* crispasr_session_beats_events(struct CrispasrSession* s, int* out_n_events);
extern int crispasr_session_beats_sample_rate(struct CrispasrSession* s);
extern float crispasr_session_beats_tempo_bpm(struct CrispasrSession* s);
extern int crispasr_session_chords(struct CrispasrSession* s, const float* pcm, int n_samples, int sample_rate);
extern const char* crispasr_session_chords_span_name(struct CrispasrSession* s, int idx);
extern const float* crispasr_session_chords_spans(struct CrispasrSession* s, int* out_n_spans);
extern int crispasr_session_chords_vocab_size(struct CrispasrSession* s);
extern int crispasr_session_convert(struct CrispasrSession* s, const float* content, int n_frames, const float* f0_hz,
                                    int speaker_id, const float* noise_zp, const float* noise_sine);
extern const float* crispasr_session_convert_audio(struct CrispasrSession* s, int* out_n_samples);
extern int crispasr_session_convert_content_dim(struct CrispasrSession* s);
extern int crispasr_session_convert_inter_channels(struct CrispasrSession* s);
extern int crispasr_session_convert_n_speakers(struct CrispasrSession* s);
extern int crispasr_session_convert_upsample_product(struct CrispasrSession* s);
extern const char* crispasr_session_disclaimer_text(void);
extern float* crispasr_session_get_disclaimer_pcm(struct CrispasrSession* s, int* out_n_samples);
extern const char* crispasr_session_last_synth_error(struct CrispasrSession* s);
extern int crispasr_session_piano(struct CrispasrSession* s, const float* pcm_16k, int n_samples);
extern const int* crispasr_session_piano_note_programs(struct CrispasrSession* s, int* out_n_notes);
extern const float* crispasr_session_piano_notes(struct CrispasrSession* s, int* out_n_notes);
extern int crispasr_session_piano_sample_rate(struct CrispasrSession* s);
extern int crispasr_session_set_tts_cfg_scale(struct CrispasrSession* s, float scale);
extern int crispasr_session_set_voice_samples(struct CrispasrSession* s, const float* pcm, int32_t n_samples,
                                              int32_t sample_rate, const char* ref_text_or_null);
extern int crispasr_session_synthesize_streaming(struct CrispasrSession* s, const char* text,
                                                 void (*cb)(const float*, int, int, void*), void* user_data);
extern int crispasr_session_tab(struct CrispasrSession* s, const float* pcm, int n_samples, int sample_rate);
extern const float* crispasr_session_tab_emissions(struct CrispasrSession* s, int* out_n_frames, int* out_n_strings,
                                                   int* out_n_classes);
extern float crispasr_session_tab_frame_period(struct CrispasrSession* s);
extern int crispasr_session_tab_silent_class(struct CrispasrSession* s);
extern int crispasr_session_tab_string_open_midi(struct CrispasrSession* s, int string);
extern int crispasr_session_tada_set_makeref_models(struct CrispasrSession* s, const char* encoder_gguf,
                                                    const char* aligner_gguf);

/* Synchronous session operations. Native views are copied before returning.
 * Music flat buffers keep the native units documented in crispasr_session.h. */
static void rb_extra_check(int rc, const char* name) {
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "%s failed (rc=%d)", name, rc);
}
static _Thread_local int rb_extra_in_stream;
// Keep deferred exceptions in Ruby's per-thread/fiber storage, so GC traces
// them. A raw C TLS VALUE is neither marked nor updated by compaction.
static ID rb_aux_error_id;
static VALUE rb_aux_pending_get(void) {
    if (!rb_aux_error_id)
        rb_aux_error_id = rb_intern("__crispasr_callback_error");
    return rb_thread_local_aref(rb_thread_current(), rb_aux_error_id);
}
#define rb_aux_pending_exc rb_aux_pending_get()
static VALUE rb_aux_remember_error(VALUE error) {
    if (NIL_P(rb_aux_pending_get()))
        rb_thread_local_aset(rb_thread_current(), rb_aux_error_id, error);
    return Qnil;
}
static void rb_aux_capture_error(void) {
    VALUE error = rb_errinfo();
    int state = 0;
    // Store the original while errinfo still roots it; no allocation failure
    // from recording an exception may unwind through native callback frames.
    rb_protect(rb_aux_remember_error, error, &state);
    if (!state)
        rb_set_errinfo(Qnil);
}
static inline void rb_aux_check_pending(void) {
    VALUE error = rb_aux_pending_get();
    if (!NIL_P(error)) {
        rb_thread_local_aset(rb_thread_current(), rb_aux_error_id, Qnil);
        rb_exc_raise(error);
    }
}
static VALUE rb_extra_sessions = Qnil, rb_extra_stream_owners = Qnil, rb_extra_active_contexts = Qnil;
static VALUE rb_parakeet_handles = Qnil, rb_titanet_handles = Qnil, rb_speaker_db_handles = Qnil;
static VALUE rb_chat_live = Qnil, rb_text_handles = Qnil, rb_extra_stream_pointers = Qnil;
static uint64_t rb_extra_ticket_counter;
static VALUE rb_extra_new_ticket(void) {
    if (rb_extra_ticket_counter == UINT64_MAX)
        rb_raise(rb_eRuntimeError, "Native handle namespace exhausted");
    return ULL2NUM(++rb_extra_ticket_counter);
}
static void* rb_extra_native_pointer(VALUE map, VALUE key, const char* kind) {
    if (!RB_INTEGER_TYPE_P(key))
        rb_raise(rb_eTypeError, "Integer native handle required");
    VALUE native = map == Qnil ? Qnil : rb_hash_aref(map, key);
    if (NIL_P(native))
        rb_raise(rb_eRuntimeError, "%s is closed or foreign", kind);
    return (void*)(uintptr_t)NUM2ULL(native);
}
struct rb_extra_owner_registration {
    VALUE* map;
    void* native;
};
static VALUE rb_extra_owner_registration_body(VALUE raw) {
    struct rb_extra_owner_registration* owner = (void*)(uintptr_t)raw;
    if (*owner->map == Qnil) {
        *owner->map = rb_hash_new();
        rb_global_variable(owner->map);
    }
    VALUE key = rb_extra_new_ticket();
    rb_hash_aset(*owner->map, key, ULL2NUM((uintptr_t)owner->native));
    return key;
}
static VALUE rb_extra_track_owner(VALUE* map, void* native, void (*release)(void*)) {
    struct rb_extra_owner_registration owner = {map, native};
    int state = 0;
    VALUE key = rb_protect(rb_extra_owner_registration_body, (VALUE)(uintptr_t)&owner, &state);
    if (state) {
        release(native);
        rb_jump_tag(state);
    }
    return key;
}

static void rb_extra_session_maps(void) {
    if (rb_extra_stream_pointers == Qnil) {
        rb_extra_stream_pointers = rb_hash_new();
        rb_global_variable(&rb_extra_stream_pointers);
    }
    if (rb_extra_active_contexts == Qnil) {
        rb_extra_active_contexts = rb_hash_new();
        rb_global_variable(&rb_extra_active_contexts);
    }
    if (rb_text_handles == Qnil) {
        rb_text_handles = rb_hash_new();
        rb_global_variable(&rb_text_handles);
    }
    if (rb_extra_sessions == Qnil) {
        rb_extra_sessions = rb_hash_new();
        rb_global_variable(&rb_extra_sessions);
    }
    if (rb_extra_stream_owners == Qnil) {
        rb_extra_stream_owners = rb_hash_new();
        rb_global_variable(&rb_extra_stream_owners);
    }
}
static void rb_extra_release_session(void* native) {
    crispasr_session_close((struct CrispasrSession*)native);
}
static VALUE rb_extra_track_session(struct CrispasrSession* native) {
    return rb_extra_track_owner(&rb_extra_sessions, native, rb_extra_release_session);
}
struct rb_extra_stream_registration {
    struct CrispasrStream* native;
    VALUE owner, key;
};
static VALUE rb_extra_stream_registration_body(VALUE raw) {
    struct rb_extra_stream_registration* registration = (void*)(uintptr_t)raw;
    VALUE key = rb_extra_new_ticket();
    registration->key = key;
    rb_hash_aset(rb_extra_stream_pointers, key, ULL2NUM((uintptr_t)registration->native));
    rb_hash_aset(rb_extra_stream_owners, key, registration->owner);
    return key;
}
static VALUE rb_extra_track_stream(struct CrispasrStream* native, VALUE owner) {
    struct rb_extra_stream_registration registration = {native, owner, Qnil};
    int state = 0;
    VALUE key = rb_protect(rb_extra_stream_registration_body, (VALUE)(uintptr_t)&registration, &state);
    if (state) {
        if (!NIL_P(registration.key))
            rb_hash_delete(rb_extra_stream_pointers, registration.key);
        crispasr_stream_close(native);
        rb_jump_tag(state);
    }
    return key;
}
static struct CrispasrStream* rb_extra_stream_handle(VALUE handle) {
    if (!RB_INTEGER_TYPE_P(handle))
        rb_raise(rb_eTypeError, "Integer stream handle required");
    if (rb_extra_in_stream)
        rb_raise(rb_eRuntimeError, "stream re-entry during callback");
    VALUE owner = rb_extra_stream_owners == Qnil ? Qnil : rb_hash_aref(rb_extra_stream_owners, handle);
    if (NIL_P(owner) || rb_extra_sessions == Qnil || NIL_P(rb_hash_aref(rb_extra_sessions, owner)))
        rb_raise(rb_eRuntimeError, "Stream is closed");
    return rb_extra_native_pointer(rb_extra_stream_pointers, handle, "Stream");
}
static struct CrispasrSession* rb_extra_handle(VALUE handle) {
    if (!RB_INTEGER_TYPE_P(handle))
        rb_raise(rb_eTypeError, "Integer session handle required");
    if (rb_extra_in_stream)
        rb_raise(rb_eRuntimeError, "session re-entry during streaming callback");
    return rb_extra_native_pointer(rb_extra_sessions, handle, "Session");
}
/* Ruby-managed scratch bytes survive exceptions without a malloc leak. */
static VALUE rb_extra_pcm(VALUE input) {
    Check_Type(input, T_ARRAY);
    long n = RARRAY_LEN(input);
    if (n <= 0 || n > INT_MAX || n > LONG_MAX / (long)sizeof(float))
        rb_raise(rb_eArgError, "PCM length out of range");
    VALUE storage = rb_str_new(NULL, n * sizeof(float));
    for (long i = 0; i < n; ++i) {
        float value = NUM2DBL(rb_ary_entry(input, i));
        ((float*)RSTRING_PTR(storage))[i] = value;
    }
    return storage;
}
static VALUE rb_extra_floats(const float* p, long n) {
    if (n < 0 || (!p && n))
        rb_raise(rb_eRuntimeError, "invalid native float buffer");
    VALUE out = rb_ary_new_capa(n);
    for (long i = 0; i < n; ++i)
        rb_ary_push(out, DBL2NUM(p[i]));
    return out;
}
#define EXTRA_SET(out, key, value) rb_hash_aset(out, ID2SYM(rb_intern(key)), value)
static VALUE rb_extra_piano(VALUE self, VALUE handle, VALUE audio) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(audio);
    int rc = crispasr_session_piano(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(audio));
    RB_GC_GUARD(pcm);
    rb_extra_check(rc, "piano");
    int n = 0, np = 0;
    const float* p = crispasr_session_piano_notes(s, &n);
    const int* programs = crispasr_session_piano_note_programs(s, &np);
    if (n < 0 || np != n || (n && !programs))
        rb_raise(rb_eRuntimeError, "invalid piano programs");
    VALUE out = rb_hash_new(), progs = rb_ary_new_capa(n);
    EXTRA_SET(out, "notes", rb_extra_floats(p, (long)n * 4));
    for (int i = 0; i < n; ++i)
        rb_ary_push(progs, INT2NUM(programs[i]));
    EXTRA_SET(out, "programs", progs);
    EXTRA_SET(out, "sample_rate", INT2NUM(crispasr_session_piano_sample_rate(s)));
    return out;
}
static VALUE rb_extra_chords(VALUE self, VALUE handle, VALUE audio, VALUE rate) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(audio);
    int rc = crispasr_session_chords(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(audio), NUM2INT(rate));
    RB_GC_GUARD(pcm);
    rb_extra_check(rc, "chords");
    int n = 0;
    const float* p = crispasr_session_chords_spans(s, &n);
    if (n < 0)
        rb_raise(rb_eRuntimeError, "invalid chord count");
    VALUE out = rb_hash_new(), names = rb_ary_new_capa(n);
    EXTRA_SET(out, "spans", rb_extra_floats(p, (long)n * 4));
    for (int i = 0; i < n; ++i) {
        const char* t = crispasr_session_chords_span_name(s, i);
        rb_ary_push(names, rb_utf8_str_new_cstr(t ? t : ""));
    }
    EXTRA_SET(out, "names", names);
    EXTRA_SET(out, "vocab_size", INT2NUM(crispasr_session_chords_vocab_size(s)));
    return out;
}
static VALUE rb_extra_beats(VALUE self, VALUE handle, VALUE audio) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(audio);
    int rate = crispasr_session_beats_sample_rate(s);
    int rc = crispasr_session_beats(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(audio), rate);
    RB_GC_GUARD(pcm);
    rb_extra_check(rc, "beats");
    int n = 0;
    const float* p = crispasr_session_beats_events(s, &n);
    if (n < 0)
        rb_raise(rb_eRuntimeError, "invalid beat count");
    VALUE out = rb_hash_new();
    EXTRA_SET(out, "events", rb_extra_floats(p, (long)n * 2));
    EXTRA_SET(out, "sample_rate", INT2NUM(rate));
    EXTRA_SET(out, "tempo_bpm", DBL2NUM(crispasr_session_beats_tempo_bpm(s)));
    return out;
}
static VALUE rb_extra_tab(VALUE self, VALUE handle, VALUE audio, VALUE rate) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(audio);
    int rc = crispasr_session_tab(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(audio), NUM2INT(rate));
    RB_GC_GUARD(pcm);
    rb_extra_check(rc, "tab");
    int n = 0, strings = 0, classes = 0;
    const float* p = crispasr_session_tab_emissions(s, &n, &strings, &classes);
    if (n < 0 || strings < 0 || classes < 0 || (strings && n > LONG_MAX / strings) ||
        (classes && (long)n * strings > LONG_MAX / classes))
        rb_raise(rb_eRuntimeError, "invalid tab dimensions");
    VALUE out = rb_hash_new(), midi = rb_ary_new_capa(strings);
    EXTRA_SET(out, "emissions", rb_extra_floats(p, (long)n * strings * classes));
    EXTRA_SET(out, "n_frames", INT2NUM(n));
    EXTRA_SET(out, "n_strings", INT2NUM(strings));
    EXTRA_SET(out, "n_classes", INT2NUM(classes));
    EXTRA_SET(out, "silent_class", INT2NUM(crispasr_session_tab_silent_class(s)));
    EXTRA_SET(out, "frame_period", DBL2NUM(crispasr_session_tab_frame_period(s)));
    for (int i = 0; i < strings; ++i)
        rb_ary_push(midi, INT2NUM(crispasr_session_tab_string_open_midi(s, i)));
    EXTRA_SET(out, "open_midi", midi);
    return out;
}
static VALUE rb_extra_convert(int argc, VALUE* argv, VALUE self) {
    VALUE handle, content, f0, speaker, zp, sine;
    rb_scan_args(argc, argv, "42", &handle, &content, &f0, &speaker, &zp, &sine);
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE c = rb_extra_pcm(content), f = rb_extra_pcm(f0), z = Qnil, v = Qnil;
    long n = RARRAY_LEN(f0);
    int dim = crispasr_session_convert_content_dim(s), id = NUM2INT(speaker);
    if (dim <= 0 || n > LONG_MAX / dim || RARRAY_LEN(content) != n * dim || id < 0 ||
        id >= crispasr_session_convert_n_speakers(s))
        rb_raise(rb_eArgError, "invalid conversion shape or speaker");
    if (!NIL_P(zp)) {
        int d = crispasr_session_convert_inter_channels(s);
        if (d <= 0 || n > LONG_MAX / d || RARRAY_LEN(zp) != n * d)
            rb_raise(rb_eArgError, "invalid noise_zp length");
        z = rb_extra_pcm(zp);
    }
    if (!NIL_P(sine)) {
        int d = crispasr_session_convert_upsample_product(s);
        if (d <= 0 || n > LONG_MAX / d || RARRAY_LEN(sine) != n * d)
            rb_raise(rb_eArgError, "invalid noise_sine length");
        v = rb_extra_pcm(sine);
    }
    int rc =
        crispasr_session_convert(s, (float*)RSTRING_PTR(c), (int)n, (float*)RSTRING_PTR(f), id,
                                 NIL_P(z) ? NULL : (float*)RSTRING_PTR(z), NIL_P(v) ? NULL : (float*)RSTRING_PTR(v));
    RB_GC_GUARD(c);
    RB_GC_GUARD(f);
    RB_GC_GUARD(z);
    RB_GC_GUARD(v);
    if (rc <= 0)
        rb_raise(rb_eRuntimeError, "convert failed (rc=%d)", rc);
    int count = 0;
    const float* p = crispasr_session_convert_audio(s, &count);
    return rb_extra_floats(p, count);
}
static VALUE rb_extra_voice_samples(VALUE self, VALUE handle, VALUE audio, VALUE rate, VALUE text) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(audio);
    int sr = NUM2INT(rate);
    if (sr <= 0)
        rb_raise(rb_eArgError, "sample rate must be positive");
    int rc = crispasr_session_set_voice_samples(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(audio), sr,
                                                NIL_P(text) ? NULL : StringValueCStr(text));
    RB_GC_GUARD(pcm);
    rb_extra_check(rc, "set_voice_samples");
    return Qnil;
}
static VALUE rb_extra_cfg(VALUE self, VALUE handle, VALUE scale) {
    rb_extra_check(crispasr_session_set_tts_cfg_scale(rb_extra_handle(handle), NUM2DBL(scale)), "set_tts_cfg_scale");
    return Qnil;
}
static VALUE rb_extra_tada(VALUE self, VALUE handle, VALUE encoder, VALUE aligner) {
    rb_extra_check(crispasr_session_tada_set_makeref_models(rb_extra_handle(handle),
                                                            NIL_P(encoder) ? NULL : StringValueCStr(encoder),
                                                            NIL_P(aligner) ? NULL : StringValueCStr(aligner)),
                   "tada_set_makeref_models");
    return Qnil;
}
static VALUE rb_extra_last_error(VALUE self, VALUE handle) {
    const char* p = crispasr_session_last_synth_error(rb_extra_handle(handle));
    return rb_utf8_str_new_cstr(p ? p : "");
}
static VALUE rb_extra_disclaimer_text(VALUE self) {
    const char* p = crispasr_session_disclaimer_text();
    return rb_utf8_str_new_cstr(p ? p : "");
}
struct rb_extra_owned_pcm {
    float* p;
    int n;
};
static VALUE rb_extra_owned_copy(VALUE ptr) {
    struct rb_extra_owned_pcm* x = (void*)(uintptr_t)ptr;
    return rb_extra_floats(x->p, x->n);
}
static VALUE rb_extra_disclaimer_pcm(VALUE self, VALUE handle) {
    struct rb_extra_owned_pcm x = {NULL, 0};
    x.p = crispasr_session_get_disclaimer_pcm(rb_extra_handle(handle), &x.n);
    int state = 0;
    VALUE out = rb_protect(rb_extra_owned_copy, (VALUE)(uintptr_t)&x, &state);
    crispasr_pcm_free(x.p);
    if (state)
        rb_jump_tag(state);
    if (!x.p)
        rb_raise(rb_eRuntimeError, "disclaimer unavailable");
    return out;
}
struct rb_extra_stream {
    const float* p;
    int n, final, state;
};
static VALUE rb_extra_yield(VALUE ptr) {
    struct rb_extra_stream* x = (void*)(uintptr_t)ptr;
    return rb_yield_values(2, rb_extra_floats(x->p, x->n), x->final ? Qtrue : Qfalse);
}
static void rb_extra_pcm_callback(const float* p, int n, int final, void* user) {
    struct rb_extra_stream* x = user;
    if (x->state)
        return;
    x->p = p;
    x->n = n;
    x->final = final;
    rb_protect(rb_extra_yield, (VALUE)(uintptr_t)x, &x->state);
}
static VALUE rb_extra_streaming(VALUE self, VALUE handle, VALUE text) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    if (!rb_block_given_p())
        rb_raise(rb_eArgError, "a block is required");
    const char* input = StringValueCStr(text);
    struct rb_extra_stream x = {0};
    rb_extra_in_stream = 1;
    int rc = crispasr_session_synthesize_streaming(s, input, rb_extra_pcm_callback, &x);
    rb_extra_in_stream = 0;
    if (x.state)
        rb_jump_tag(x.state);
    rb_extra_check(rc, "synthesize_streaming");
    return Qnil;
}

#include "ruby_crispasr_context_guard.h"
