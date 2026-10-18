// TitaNet transport oracle: runtime contract is 192 floats written or 0 error.
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int titan_mode, titan_live, titan_calls, titan_reuse;
static int titan_reused_context;
void probe_titanet_force_reuse(int enabled) {
    titan_reuse = enabled;
}
void probe_titanet_mode(int mode) {
    titan_mode = mode;
}
int probe_titanet_live(void) {
    return titan_live;
}
int probe_titanet_calls(void) {
    return titan_calls;
}
void* crispasr_titanet_init(const char* model, int32_t threads) {
    if (!strcmp(model, "fail"))
        return NULL;
    ++titan_live;
    return titan_reuse ? (void*)&titan_reused_context : malloc(1);
}
void crispasr_titanet_free(void* ctx) {
    if (ctx) {
        --titan_live;
        if (ctx != &titan_reused_context)
            free(ctx);
    }
}
int32_t crispasr_titanet_embed(void* ctx, const float* pcm, int32_t count, float* out) {
    ++titan_calls;
    if (!ctx || !pcm || count != 3 || pcm[0] != 1 || pcm[2] != 5 || titan_mode == 2)
        return 0;
    for (int i = 0; i < 192; ++i)
        out[i] = i == 0 ? 1.f : 0.f;
    return titan_mode == 3 ? 193 : titan_mode == 4 ? 1 : 192;
}
float crispasr_titanet_cosine_sim(const float* a, const float* b, int32_t dim) {
    float dot = 0;
    for (int i = 0; i < dim; ++i)
        dot += a[i] * b[i];
    return dot;
}
