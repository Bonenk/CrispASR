// Public speaker primitives. Typed Ruby owners free native contexts even on exceptions.
struct rb_speaker_owner {
    void* handle;
    int kind; // 1=generic embedder, 2=pyannote cache
};
static void rb_speaker_owner_free(void* raw) {
    struct rb_speaker_owner* owner = raw;
    if (!owner)
        return;
    if (owner->handle) {
        if (owner->kind == 1)
            crispasr_speaker_embedder_free_abi(owner->handle);
        else
            crispasr_pyannote_cache_free_abi(owner->handle);
    }
    xfree(owner);
}
static size_t rb_speaker_owner_size(const void* raw) {
    return sizeof(struct rb_speaker_owner);
}
static const rb_data_type_t rb_speaker_owner_type = {
    .wrap_struct_name = "CrispASR.speaker_owner",
    .function = {.dfree = rb_speaker_owner_free, .dsize = rb_speaker_owner_size},
    .flags = RUBY_TYPED_FREE_IMMEDIATELY,
};
static VALUE rb_speaker_owner_alloc(VALUE klass) {
    struct rb_speaker_owner* owner;
    VALUE out = TypedData_Make_Struct(klass, struct rb_speaker_owner, &rb_speaker_owner_type, owner);
    owner->handle = NULL;
    owner->kind = 0;
    return out;
}
static struct rb_speaker_owner* rb_speaker_owner_get(VALUE self, int live) {
    struct rb_speaker_owner* owner;
    TypedData_Get_Struct(self, struct rb_speaker_owner, &rb_speaker_owner_type, owner);
    if (live && !owner->handle)
        rb_raise(rb_eRuntimeError, "Speaker context is closed");
    return owner;
}
static VALUE rb_speaker_close(VALUE self) {
    struct rb_speaker_owner* owner = rb_speaker_owner_get(self, 0);
    if (owner->handle) {
        if (owner->kind == 1)
            crispasr_speaker_embedder_free_abi(owner->handle);
        else
            crispasr_pyannote_cache_free_abi(owner->handle);
        owner->handle = NULL;
    }
    return Qnil;
}
static VALUE rb_embedder_initialize(int argc, VALUE* argv, VALUE self) {
    VALUE model, threads, cache;
    rb_scan_args(argc, argv, "12", &model, &threads, &cache);
    struct rb_speaker_owner* owner = rb_speaker_owner_get(self, 0);
    if (owner->handle)
        rb_raise(rb_eRuntimeError, "Speaker context already initialized");
    int nt = NIL_P(threads) ? 4 : NUM2INT(threads);
    if (nt <= 0)
        rb_raise(rb_eArgError, "Positive thread count required");
    VALUE roots = rb_ary_new();
    const char* path = rb_chat_owned_cstring(model, roots);
    const char* dir = NIL_P(cache) ? NULL : rb_chat_owned_cstring(cache, roots);
    if (owner->handle)
        rb_raise(rb_eRuntimeError, "Speaker context initialized during argument coercion");
    owner->kind = 1;
    owner->handle = crispasr_speaker_embedder_make_abi(path, nt, dir);
    RB_GC_GUARD(roots);
    if (!owner->handle)
        rb_raise(rb_eRuntimeError, "speaker_embedder_make failed");
    if (crispasr_speaker_embedder_dim_abi(owner->handle) <= 0) {
        rb_speaker_close(self);
        rb_raise(rb_eRuntimeError, "Invalid native embedding dimension");
    }
    return self;
}
static VALUE rb_embedder_dimension(VALUE self) {
    return INT2NUM(crispasr_speaker_embedder_dim_abi(rb_speaker_owner_get(self, 1)->handle));
}
static VALUE rb_embedder_name(VALUE self) {
    const char* name = crispasr_speaker_embedder_name_abi(rb_speaker_owner_get(self, 1)->handle);
    return rb_utf8_str_new_cstr(name ? name : "");
}
static VALUE rb_embedder_embed(VALUE self, VALUE audio) {
    struct rb_speaker_owner* owner = rb_speaker_owner_get(self, 1);
    VALUE pcm = rb_aux_pcm_storage(audio);
    long n = RARRAY_LEN(audio);
    if (!n)
        rb_raise(rb_eArgError, "Nonempty PCM required");
    int dim = crispasr_speaker_embedder_dim_abi(owner->handle);
    if (dim <= 0 || dim > LONG_MAX / (long)sizeof(float))
        rb_raise(rb_eRuntimeError, "Invalid native embedding dimension");
    VALUE storage = rb_chat_buffer((long)dim * sizeof(float));
    float* output = DATA_PTR(storage);
    int rc = crispasr_speaker_embedder_embed_abi(owner->handle, (float*)RSTRING_PTR(pcm), n, output);
    RB_GC_GUARD(pcm);
    if (rc != 1)
        rb_raise(rb_eRuntimeError, "speaker_embedder_embed failed");
    VALUE out = rb_ary_new_capa(dim);
    for (int i = 0; i < dim; ++i)
        rb_ary_push(out, DBL2NUM(output[i]));
    RB_GC_GUARD(storage);
    return out;
}
static VALUE rb_speaker_cluster(int argc, VALUE* argv, VALUE self) {
    VALUE rows, threshold, maximum;
    rb_scan_args(argc, argv, "12", &rows, &threshold, &maximum);
    Check_Type(rows, T_ARRAY);
    long n = RARRAY_LEN(rows);
    if (!n || n > INT_MAX)
        rb_raise(rb_eArgError, "Nonempty embedding matrix required");
    VALUE first = rb_ary_entry(rows, 0);
    Check_Type(first, T_ARRAY);
    long dim = RARRAY_LEN(first);
    if (!dim || dim > INT_MAX || n > LONG_MAX / (long)sizeof(float) / dim)
        rb_raise(rb_eArgError, "Invalid embedding dimensions");
    float limit = NIL_P(threshold) ? .7f : NUM2DBL(threshold);
    int max = NIL_P(maximum) ? 8 : NUM2INT(maximum);
    if (!isfinite(limit) || max < 0)
        rb_raise(rb_eArgError, "Invalid clustering threshold or maximum");
    VALUE matrix = rb_chat_buffer(n * dim * sizeof(float));
    float* native = DATA_PTR(matrix);
    for (long i = 0; i < n; ++i) {
        VALUE row = rb_ary_entry(rows, i);
        Check_Type(row, T_ARRAY);
        if (RARRAY_LEN(row) != dim)
            rb_raise(rb_eArgError, "Embedding dimensions must match");
        for (long j = 0; j < dim; ++j)
            native[i * dim + j] = NUM2DBL(rb_ary_entry(row, j));
    }
    VALUE labels = rb_chat_buffer(n * sizeof(int32_t));
    int32_t* output = DATA_PTR(labels);
    int clusters = crispasr_speaker_cluster_abi(native, n, dim, limit, max, output);
    RB_GC_GUARD(matrix);
    if (clusters < 0 || clusters > n)
        rb_raise(rb_eRuntimeError, "speaker_cluster failed");
    VALUE result = rb_hash_new(), values = rb_ary_new_capa(n);
    for (long i = 0; i < n; ++i)
        rb_ary_push(values, INT2NUM(output[i]));
    rb_hash_aset(result, ID2SYM(rb_intern("labels")), values);
    rb_hash_aset(result, ID2SYM(rb_intern("n_clusters")), INT2NUM(clusters));
    RB_GC_GUARD(labels);
    return result;
}
static VALUE rb_pycache_initialize(int argc, VALUE* argv, VALUE self) {
    VALUE audio, model, threads;
    rb_scan_args(argc, argv, "21", &audio, &model, &threads);
    struct rb_speaker_owner* owner = rb_speaker_owner_get(self, 0);
    if (owner->handle)
        rb_raise(rb_eRuntimeError, "Speaker context already initialized");
    int nt = NIL_P(threads) ? 4 : NUM2INT(threads);
    if (nt <= 0)
        rb_raise(rb_eArgError, "Positive thread count required");
    VALUE pcm = rb_aux_pcm_storage(audio);
    if (!RARRAY_LEN(audio))
        rb_raise(rb_eArgError, "Nonempty PCM required");
    VALUE roots = rb_ary_new();
    const char* path = rb_chat_owned_cstring(model, roots);
    if (owner->handle)
        rb_raise(rb_eRuntimeError, "Speaker context initialized during argument coercion");
    owner->kind = 2;
    owner->handle = crispasr_pyannote_cache_compute_abi((float*)RSTRING_PTR(pcm), RARRAY_LEN(audio), path, nt);
    RB_GC_GUARD(pcm);
    RB_GC_GUARD(roots);
    if (!owner->handle)
        rb_raise(rb_eRuntimeError, "pyannote_cache_compute failed");
    return self;
}
static VALUE rb_pycache_apply(int argc, VALUE* argv, VALUE self) {
    VALUE segments, offset;
    rb_scan_args(argc, argv, "11", &segments, &offset);
    struct rb_speaker_owner* owner = rb_speaker_owner_get(self, 1);
    Check_Type(segments, T_ARRAY);
    long n = RARRAY_LEN(segments);
    if (!n || n > INT_MAX || n > LONG_MAX / (long)sizeof(struct rb_aux_diarize_segment))
        rb_raise(rb_eArgError, "Nonempty segment array required");
    int64_t t0 = NIL_P(offset) ? 0 : NUM2LL(offset);
    VALUE storage = rb_chat_buffer(n * sizeof(struct rb_aux_diarize_segment));
    struct rb_aux_diarize_segment* native = DATA_PTR(storage);
    for (long i = 0; i < n; ++i) {
        VALUE item = rb_ary_entry(segments, i);
        Check_Type(item, T_HASH);
        native[i].t0_cs = NUM2LL(rb_aux_hash_field(item, "t0"));
        native[i].t1_cs = NUM2LL(rb_aux_hash_field(item, "t1"));
        if (native[i].t0_cs < 0 || native[i].t1_cs < native[i].t0_cs)
            rb_raise(rb_eArgError, "Invalid segment times");
        native[i].speaker = -1;
    }
    int rc = crispasr_pyannote_cache_apply_abi(owner->handle, t0, native, n);
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "pyannote_cache_apply failed");
    VALUE out = rb_aux_diarize_result(native, n);
    RB_GC_GUARD(storage);
    return out;
}
static void rb_speaker_aux_init(void) {
    VALUE embedder = rb_define_class_under(mCrispASR, "SpeakerEmbedder", rb_cObject);
    rb_define_alloc_func(embedder, rb_speaker_owner_alloc);
    rb_define_method(embedder, "initialize", rb_embedder_initialize, -1);
    rb_define_method(embedder, "dimension", rb_embedder_dimension, 0);
    rb_define_method(embedder, "name", rb_embedder_name, 0);
    rb_define_method(embedder, "embed", rb_embedder_embed, 1);
    rb_define_method(embedder, "close", rb_speaker_close, 0);
    VALUE cache = rb_define_class_under(mCrispASR, "PyannoteCache", rb_cObject);
    rb_define_alloc_func(cache, rb_speaker_owner_alloc);
    rb_define_method(cache, "initialize", rb_pycache_initialize, -1);
    rb_define_method(cache, "apply", rb_pycache_apply, -1);
    rb_define_method(cache, "close", rb_speaker_close, 0);
    rb_define_singleton_method(mCrispASR, "speaker_cluster", rb_speaker_cluster, -1);
}
