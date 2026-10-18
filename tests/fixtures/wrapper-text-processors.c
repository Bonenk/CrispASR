// Prefix-specific ownership oracle for standalone processors.
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int text_mode, text_live, text_contexts;
void probe_text_mode(int mode) {
    text_mode = mode;
}
int probe_text_live(void) {
    return text_live;
}
int probe_text_contexts(void) {
    return text_contexts;
}
static void* text_init(const char* model, int kind) {
    if (!model || strcmp(model, "model") || text_mode == 3)
        return NULL;
    int* handle = malloc(sizeof(int));
    if (handle) {
        *handle = kind;
        ++text_contexts;
    }
    return handle;
}
static void text_close(void* handle, int kind) {
    if (handle) {
        if (*(int*)handle != kind)
            abort();
        --text_contexts;
        free(handle);
    }
}
static const char* text_process(void* handle, const char* input, int kind) {
    if (!handle || *(int*)handle != kind || !input)
        abort();
    if (text_mode == 1)
        return NULL;
    const char* text = text_mode == 2 ? ""
                       : kind == 1    ? "Gr\xc3\xbc\xc3\x9f"
                                        "e"
                       : kind == 2    ? "Hello."
                                      : "Hello!";
    size_t size = strlen(text) + 1;
    int* block = malloc(sizeof(int) + size);
    if (!block)
        abort();
    *block = kind;
    char* output = (char*)(block + 1);
    memcpy(output, text, size);
    ++text_live;
    return output;
}
static void text_release(const char* text, int kind) {
    if (text) {
        int* block = (int*)text - 1;
        if (*block != kind)
            abort();
        --text_live;
        free(block);
    }
}
#define DEFINE_TEXT(prefix, kind)                                                                                      \
    void* crispasr_##prefix##_init(const char* model) {                                                                \
        return text_init(model, kind);                                                                                 \
    }                                                                                                                  \
    const char* crispasr_##prefix##_process(void* handle, const char* text) {                                          \
        return text_process(handle, text, kind);                                                                       \
    }                                                                                                                  \
    void crispasr_##prefix##_free_text(const char* text) {                                                             \
        text_release(text, kind);                                                                                      \
    }                                                                                                                  \
    void crispasr_##prefix##_free(void* handle) {                                                                      \
        text_close(handle, kind);                                                                                      \
    }
DEFINE_TEXT(truecase, 1)
DEFINE_TEXT(pcs, 2)
DEFINE_TEXT(punc, 3)
