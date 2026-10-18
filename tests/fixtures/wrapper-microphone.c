// Real native producer thread, model/device-free PCM transport fixture.
#include "crispasr_mic.h"
#include <pthread.h>
#include <stdlib.h>
struct crispasr_mic {
    crispasr_mic_callback callback;
    void* user;
};
static int mic_mode, mic_live, mic_closes;
void probe_mic_mode(int mode) {
    mic_mode = mode;
}
int probe_mic_live(void) {
    return mic_live;
}
int probe_mic_copy_failure(void) {
    return mic_mode == 5;
}
int probe_mic_closes(void) {
    return mic_closes;
}
struct crispasr_mic* crispasr_mic_open(int rate, int channels, crispasr_mic_callback callback, void* user) {
    if (mic_mode == 1 || rate != 16000 || channels != 1)
        return NULL;
    struct crispasr_mic* m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->callback = callback;
    m->user = user;
    ++mic_live;
    return m;
}
static void* produce(void* raw) {
    struct crispasr_mic* m = raw;
    int n = mic_mode == 4 ? 70 : 1;
    for (int i = 0; i < n; ++i) {
        float audio[] = {(float)i, .5f, -.25f};
        m->callback(audio, 3, m->user);
    }
    return NULL;
}
int crispasr_mic_start(struct crispasr_mic* m) {
    if (mic_mode == 2)
        return 1;
    pthread_t worker;
    if (pthread_create(&worker, NULL, produce, m))
        return 1;
    pthread_join(worker, NULL);
    return 0;
}
int crispasr_mic_stop(struct crispasr_mic* m) {
    return mic_mode == 3 ? 1 : 0;
}
void crispasr_mic_close(struct crispasr_mic* m) {
    if (m) {
        --mic_live;
        ++mic_closes;
        free(m);
    }
}
const char* crispasr_mic_default_device_name(void) {
    return "Probe microphone";
}
