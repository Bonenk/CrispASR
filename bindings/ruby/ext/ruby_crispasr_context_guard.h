#pragma once
// A per-context borrow survives Ruby argument coercion, callbacks and thread/fiber switches.
// Same-context re-entry raises before native state can be closed or reused.
struct rb_context_call {
    VALUE (*fn)(ANYARGS);
    int arity, argc;
    VALUE self;
    VALUE* argv;
    VALUE scope;
    int entered;
};
static int rb_context_contains(VALUE map, VALUE key) {
    return map != Qnil && !NIL_P(rb_hash_aref(map, key));
}
static VALUE rb_context_scope(int argc, VALUE* argv) {
    if (argc < 1 || !RB_INTEGER_TYPE_P(argv[0]))
        return Qnil;
    VALUE key = argv[0];
    if (rb_context_contains(rb_extra_sessions, key))
        return key;
    if (rb_extra_stream_owners != Qnil) {
        VALUE owner = rb_hash_aref(rb_extra_stream_owners, key);
        if (!NIL_P(owner))
            return owner;
    }
    if (rb_context_contains(rb_parakeet_handles, key) || rb_context_contains(rb_titanet_handles, key) ||
        rb_context_contains(rb_speaker_db_handles, key) || rb_context_contains(rb_chat_live, key) ||
        rb_context_contains(rb_text_handles, key))
        return key;
    return Qnil;
}
static VALUE rb_context_call_body(VALUE raw) {
    struct rb_context_call* call = (void*)(uintptr_t)raw;
    if (!NIL_P(call->scope)) {
        rb_extra_session_maps();
        if (rb_context_contains(rb_extra_active_contexts, call->scope))
            rb_raise(rb_eRuntimeError, "Native context re-entry during an active operation");
        rb_hash_aset(rb_extra_active_contexts, call->scope, Qtrue);
        call->entered = 1;
    }
    VALUE* a = call->argv;
    switch (call->arity) {
    case -1:
        return ((VALUE(*)(int, VALUE*, VALUE))call->fn)(call->argc, a, call->self);
    case 0:
        return ((VALUE(*)(VALUE))call->fn)(call->self);
    case 1:
        return ((VALUE(*)(VALUE, VALUE))call->fn)(call->self, a[0]);
    case 2:
        return ((VALUE(*)(VALUE, VALUE, VALUE))call->fn)(call->self, a[0], a[1]);
    case 3:
        return ((VALUE(*)(VALUE, VALUE, VALUE, VALUE))call->fn)(call->self, a[0], a[1], a[2]);
    case 4:
        return ((VALUE(*)(VALUE, VALUE, VALUE, VALUE, VALUE))call->fn)(call->self, a[0], a[1], a[2], a[3]);
    case 5:
        return ((VALUE(*)(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE))call->fn)(call->self, a[0], a[1], a[2], a[3], a[4]);
    case 6:
        return ((VALUE(*)(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE))call->fn)(call->self, a[0], a[1], a[2], a[3],
                                                                                     a[4], a[5]);
    default:
        rb_raise(rb_eRuntimeError, "Unsupported native wrapper arity");
    }
}
static VALUE rb_context_call_cleanup(VALUE raw) {
    struct rb_context_call* call = (void*)(uintptr_t)raw;
    if (call->entered)
        rb_hash_delete(rb_extra_active_contexts, call->scope);
    return Qnil;
}
static VALUE rb_context_run(VALUE (*fn)(ANYARGS), int arity, int argc, VALUE* argv, VALUE self) {
    struct rb_context_call call = {fn, arity, argc, self, argv, rb_context_scope(argc, argv), 0};
    return rb_ensure(rb_context_call_body, (VALUE)(uintptr_t)&call, rb_context_call_cleanup, (VALUE)(uintptr_t)&call);
}
#define RB_CONTEXT_WRAPPER_VAR(fn)                                                                                     \
    static VALUE rb_context_guard_##fn(int argc, VALUE* argv, VALUE self) {                                            \
        return rb_context_run(RUBY_METHOD_FUNC(fn), -1, argc, argv, self);                                             \
    }
#define RB_CONTEXT_WRAPPER_0(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self) {                                                                   \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 0, 0, NULL, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_1(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0) {                                                         \
        VALUE argv[] = {a0};                                                                                           \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 1, 1, argv, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_2(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0, VALUE a1) {                                               \
        VALUE argv[] = {a0, a1};                                                                                       \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 2, 2, argv, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_3(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0, VALUE a1, VALUE a2) {                                     \
        VALUE argv[] = {a0, a1, a2};                                                                                   \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 3, 3, argv, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_4(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0, VALUE a1, VALUE a2, VALUE a3) {                           \
        VALUE argv[] = {a0, a1, a2, a3};                                                                               \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 4, 4, argv, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_5(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0, VALUE a1, VALUE a2, VALUE a3, VALUE a4) {                 \
        VALUE argv[] = {a0, a1, a2, a3, a4};                                                                           \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 5, 5, argv, self);                                                 \
    }
#define RB_CONTEXT_WRAPPER_6(fn)                                                                                       \
    static VALUE rb_context_guard_##fn(VALUE self, VALUE a0, VALUE a1, VALUE a2, VALUE a3, VALUE a4, VALUE a5) {       \
        VALUE argv[] = {a0, a1, a2, a3, a4, a5};                                                                       \
        return rb_context_run(RUBY_METHOD_FUNC(fn), 6, 6, argv, self);                                                 \
    }
