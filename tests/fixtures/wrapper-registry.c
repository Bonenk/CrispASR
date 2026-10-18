// Model-free registry/cache transport oracle. No downloads or filesystem work.
#include "crispasr_session.h"
#include <string.h>
static int registry_mode;
void probe_registry_mode(int mode) {
    registry_mode = mode;
}
static int copy_field(const char* text, char* out, int cap) {
    if (registry_mode == 2)
        return -1;
    if (registry_mode == 4 || (registry_mode == 3 && cap < 512))
        return 2;
    if ((int)strlen(text) >= cap)
        return 2;
    if (registry_mode == 5) {
        memset(out, 'x', cap);
        return 0;
    }
    memcpy(out, text, strlen(text) + 1);
    return 0;
}
static int entry(const char* file, char* f, int fc, char* u, int uc, char* s, int sc) {
    int rc = copy_field(file, f, fc);
    if (rc)
        return rc;
    rc = copy_field("https://example.invalid/model", u, uc);
    if (rc)
        return rc;
    return copy_field("715 MB", s, sc);
}
int crispasr_registry_lookup_abi(const char* b, char* f, int32_t fc, char* u, int32_t uc, char* s, int32_t sc) {
    if (strcmp(b, "foo"))
        return 1;
    return entry("foo.gguf", f, fc, u, uc, s, sc);
}
int crispasr_registry_lookup_by_filename_abi(const char* key, char* f, int32_t fc, char* u, int32_t uc, char* s,
                                             int32_t sc) {
    if (strcmp(key, "foo.gguf"))
        return 1;
    return entry("foo.gguf", f, fc, u, uc, s, sc);
}
int crispasr_registry_list_backends_abi(char* out, int32_t cap) {
    int rc = copy_field("foo,empty", out, cap);
    return rc == 2 ? -2 : rc ? rc : 9;
}
int crispasr_registry_default_bundle_info_abi(const char* b, char* out, int32_t cap, char* license, int32_t lc,
                                              int32_t* accept) {
    if (strcmp(b, "foo") && strcmp(b, "empty"))
        return 0;
    int rc = copy_field(b, out, cap);
    if (rc)
        return rc == 2 ? -2 : rc;
    rc = copy_field("Apache-2.0", license, lc);
    if (rc)
        return rc == 2 ? -2 : rc;
    *accept = 1;
    return strcmp(b, "empty") ? 3 : 0;
}
int crispasr_registry_default_bundle_artifact_abi(const char* b, int32_t i, int32_t* kind, char* f, int32_t fc, char* u,
                                                  int32_t uc, char* s, int32_t sc) {
    if (strcmp(b, "foo") || i < 0 || i > 2 || registry_mode == 6)
        return 1;
    *kind = registry_mode == 7 ? 9 : i;
    return entry(i == 0 ? "foo.gguf" : i == 1 ? "codec.gguf" : "tokens.json", f, fc, u, uc, s, sc);
}
int crispasr_cache_dir_abi(const char* override_dir, char* out, int32_t cap) {
    if (override_dir && strcmp(override_dir, "/cache"))
        return -1;
    return copy_field(override_dir ? "/cache" : "/default-cache", out, cap);
}
int crispasr_cache_ensure_file_abi(const char* file, const char* url, int32_t quiet, const char* override_dir,
                                   char* out, int32_t cap) {
    if (strcmp(file, "foo.gguf") || strcmp(url, "https://example.invalid/model") || quiet != 1 || !override_dir ||
        strcmp(override_dir, "/cache"))
        return 1;
    return copy_field("/cache/foo.gguf", out, cap);
}
