// crispasr_c2pa.h — C2PA (Content Credentials) manifest signing for TTS output.
//
// Compile-time gated on CRISPASR_HAVE_C2PA. When enabled, signs synthesized
// audio (WAV / MP3 / M4A / FLAC — whatever the c2pa runtime supports) with a
// C2PA manifest declaring AI-generated provenance, in memory (no temp files).
//
// When CRISPASR_HAVE_C2PA is not defined, the functions are no-ops that return
// false and leave the buffer unchanged (the watermark + container metadata tag
// still provide provenance).
//
// Targets the c2pa-rs C ABI (c2pa.h from the prebuilt native lib, v0.89+):
//   c2pa_signer_from_info() / c2pa_builder_from_json() / c2pa_builder_sign()
//   with callback-based streams (c2pa_create_stream). Sign in memory over a
//   std::string buffer for any supported format.
//
// Certificate: a self-signed X.509 (P-256 / ES256) is sufficient for
// machine-readable AI marking (EU AI Act Art. 50). C2PA verifiers show
// "unverified signer" for self-signed certs; the manifest is still valid.
// Generate with scripts/generate-c2pa-cert.sh, or let CrispASR auto-provision a
// per-install self-signed cert (see crispasr_c2pa_autocert() below).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>

#ifdef CRISPASR_HAVE_C2PA
#include <c2pa.h>
#endif

// The C2PA manifest JSON. digitalSourceType per IPTC vocabulary marks this as
// trained-algorithmic (AI-generated) media.
inline const char* crispasr_c2pa_manifest_json() {
    return R"({
  "claim_generator": "CrispASR",
  "claim_generator_info": [{
    "name": "CrispASR",
    "version": "0.6"
  }],
  "assertions": [{
    "label": "c2pa.actions",
    "data": {
      "actions": [{
        "action": "c2pa.created",
        "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "softwareAgent": "CrispASR TTS"
      }]
    }
  }]
})";
}

// Map an output file extension (lowercased, no dot) to the C2PA MIME/format
// string, or "" if c2pa cannot embed a manifest in that container. c2pa-rs
// supports WAV/MP3/M4A/MP4/FLAC for audio; it does NOT support Ogg/Opus or raw
// ADTS AAC, so those return "" (caller skips signing, keeps watermark + tag).
inline const char* crispasr_c2pa_format_for_ext(const std::string& ext) {
    if (ext == "wav")
        return "audio/wav";
    if (ext == "mp3")
        return "audio/mpeg";
    if (ext == "m4a")
        return "audio/mp4";
    if (ext == "mp4")
        return "audio/mp4";
    if (ext == "flac")
        return "audio/flac";
    return ""; // aac (adts), opus/ogg — not embeddable by c2pa
}

#ifdef CRISPASR_HAVE_C2PA

namespace crispasr_c2pa_detail {

// In-memory stream backing for the c2pa callback stream API. `buf` is owned by
// the caller; `pos` is the stream cursor.
struct membuf {
    std::string* buf;
    size_t pos;
};

inline intptr_t mem_read(StreamContext* ctx, uint8_t* data, intptr_t len) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    if (len < 0)
        return -1;
    size_t avail = m->buf->size() > m->pos ? m->buf->size() - m->pos : 0;
    size_t n = std::min(static_cast<size_t>(len), avail);
    if (n)
        std::memcpy(data, m->buf->data() + m->pos, n);
    m->pos += n;
    return static_cast<intptr_t>(n);
}

inline intptr_t mem_seek(StreamContext* ctx, intptr_t offset, C2paSeekMode mode) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    intptr_t base = (mode == Start)     ? 0
                    : (mode == Current) ? static_cast<intptr_t>(m->pos)
                                        : static_cast<intptr_t>(m->buf->size());
    intptr_t np = base + offset;
    if (np < 0)
        np = 0;
    m->pos = static_cast<size_t>(np);
    return static_cast<intptr_t>(m->pos);
}

inline intptr_t mem_write(StreamContext* ctx, const uint8_t* data, intptr_t len) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    if (len < 0)
        return -1;
    if (m->pos + static_cast<size_t>(len) > m->buf->size())
        m->buf->resize(m->pos + static_cast<size_t>(len));
    if (len)
        std::memcpy(&(*m->buf)[m->pos], data, static_cast<size_t>(len));
    m->pos += static_cast<size_t>(len);
    return len;
}

inline intptr_t mem_flush(StreamContext*) {
    return 0;
}

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace crispasr_c2pa_detail

#endif // CRISPASR_HAVE_C2PA

// Sign an in-memory audio buffer with a C2PA manifest. `format` is a C2PA
// MIME/format string (see crispasr_c2pa_format_for_ext). `cert_pem`/`key_pem`
// are PATHS to PEM files. On success, `data` is replaced with the signed asset
// and true is returned. Returns false (leaving `data` unchanged) when C2PA is
// unavailable, inputs are missing, the format is unsupported, or signing fails.
inline bool crispasr_c2pa_sign_buffer(std::string& data, const char* format, const std::string& cert_pem,
                                      const std::string& key_pem) {
    if (!format || !*format || cert_pem.empty() || key_pem.empty())
        return false;

#ifdef CRISPASR_HAVE_C2PA
    using namespace crispasr_c2pa_detail;

    // c2pa 0.89 marks c2pa_builder_from_json / c2pa_signer_free deprecated in
    // favor of newer context/free APIs; the deprecated ones remain functional
    // and are the stable path across the prebuilt lib versions we pin. Silence
    // the warning locally so it doesn't trip a -Werror build.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    std::string cert = read_file(cert_pem);
    std::string key = read_file(key_pem);
    if (cert.empty() || key.empty()) {
        fprintf(stderr, "crispasr: C2PA cert/key unreadable ('%s' / '%s')\n", cert_pem.c_str(), key_pem.c_str());
        return false;
    }

    // Self-signed P-256 EC cert from generate-c2pa-cert.sh → ES256.
    C2paSignerInfo info;
    info.alg = "es256";
    info.sign_cert = cert.c_str();
    info.private_key = key.c_str();
    info.ta_url = nullptr;
    C2paSigner* signer = c2pa_signer_from_info(&info);
    if (!signer) {
        fprintf(stderr, "crispasr: C2PA signer init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        return false;
    }

    C2paBuilder* builder = c2pa_builder_from_json(crispasr_c2pa_manifest_json());
    if (!builder) {
        fprintf(stderr, "crispasr: C2PA builder init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        c2pa_signer_free(signer);
        return false;
    }

    membuf src{&data, 0};
    std::string out;
    membuf dst{&out, 0};
    C2paStream* ss =
        c2pa_create_stream(reinterpret_cast<StreamContext*>(&src), mem_read, mem_seek, mem_write, mem_flush);
    C2paStream* ds =
        c2pa_create_stream(reinterpret_cast<StreamContext*>(&dst), mem_read, mem_seek, mem_write, mem_flush);

    const unsigned char* manifest_bytes = nullptr;
    int64_t rc = c2pa_builder_sign(builder, format, ss, ds, signer, &manifest_bytes);
    if (manifest_bytes)
        c2pa_manifest_bytes_free(manifest_bytes);
    c2pa_release_stream(ss);
    c2pa_release_stream(ds);
    c2pa_builder_free(builder);
    c2pa_signer_free(signer);

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    if (rc < 0) {
        fprintf(stderr, "crispasr: C2PA signing failed (%s): %s\n", format, c2pa_error() ? c2pa_error() : "unknown");
        return false;
    }
    data.swap(out);
    return true;
#else
    (void)data;
    (void)cert_pem;
    (void)key_pem;
    return false;
#endif
}

// Back-compat WAV wrapper (existing call sites). Prefer crispasr_c2pa_sign_buffer.
inline bool crispasr_c2pa_sign_wav(std::string& wav, const std::string& cert_pem, const std::string& key_pem) {
    return crispasr_c2pa_sign_buffer(wav, "audio/wav", cert_pem, key_pem);
}

inline bool crispasr_c2pa_autocert(const std::string& cache_dir, std::string& out_cert, std::string& out_key);

// Sign `data` (format = C2PA MIME, e.g. "audio/wav" / "audio/mpeg") with the
// EFFECTIVE credentials: the user's --c2pa-cert/--c2pa-key if given, otherwise an
// auto-provisioned per-install self-signed cert (on-by-default provenance).
// Best-effort — returns false (leaving `data` unchanged) if C2PA is unavailable,
// the format is empty/unsupported, cert provisioning fails, or signing fails.
inline bool crispasr_c2pa_sign_auto(std::string& data, const char* format, const std::string& user_cert,
                                    const std::string& user_key, const std::string& cache_dir) {
    if (!format || !*format)
        return false;
    std::string cert = user_cert, key = user_key;
    if (cert.empty() || key.empty()) {
#ifdef CRISPASR_HAVE_C2PA
        if (!crispasr_c2pa_autocert(cache_dir, cert, key))
            return false;
#else
        (void)cache_dir;
        return false;
#endif
    }
    return crispasr_c2pa_sign_buffer(data, format, cert, key);
}

// Ensure a per-install self-signed C2PA certificate exists, for on-by-default
// signing when the user didn't pass --c2pa-cert. Generates one with `openssl`
// into <cache_dir>/c2pa/ on first use and caches it (10-year P-256 / ES256, with
// the C2PA-required extensions). Returns true and fills cert/key paths on
// success; false if openssl is unavailable or generation fails (the caller then
// falls back to watermark-only — C2PA is a best-effort upgrade, never fatal).
//
// A self-signed cert marks content as AI-generated in a machine-readable way
// (EU AI Act Art. 50); verifiers show "unverified signer". For a trusted signer
// identity, pass a CA-issued cert via --c2pa-cert / --c2pa-key instead.
inline bool crispasr_c2pa_autocert(const std::string& cache_dir, std::string& out_cert, std::string& out_key) {
#if defined(__EMSCRIPTEN__)
    // No usable process/openssl in the browser sandbox — on-by-default self-sign
    // isn't possible here. The host must pass a cert/key (or a build-time bundled
    // cert) to sign in WASM; otherwise C2PA is skipped (watermark still applies).
    (void)cache_dir;
    (void)out_cert;
    (void)out_key;
    return false;
#else
    namespace fs = std::filesystem;
    std::string base = cache_dir;
    if (base.empty()) {
        const char* home = std::getenv("HOME");
#ifdef _WIN32
        if (!home)
            home = std::getenv("USERPROFILE");
#endif
        if (!home)
            return false;
        base = std::string(home) + "/.cache/crispasr";
    }
    const std::string dir = base + "/c2pa";
    const std::string cert = dir + "/crispasr-c2pa.crt";
    const std::string key = dir + "/crispasr-c2pa.key";

    std::error_code ec;
    if (fs::exists(cert, ec) && fs::exists(key, ec)) {
        out_cert = cert;
        out_key = key;
        return true;
    }
    fs::create_directories(dir, ec);
    if (ec)
        return false;

    const std::string cnf = dir + "/c2pa-ext.cnf";
    {
        std::ofstream f(cnf);
        if (!f)
            return false;
        f << "[req]\ndistinguished_name = dn\nx509_extensions = v3\nprompt = no\n"
             "[dn]\nCN = CrispASR TTS\nO = Self-Signed C2PA\n"
             "[v3]\nbasicConstraints = critical, CA:FALSE\n"
             "keyUsage = critical, digitalSignature\n"
             "extendedKeyUsage = critical, emailProtection\n"
             "subjectKeyIdentifier = hash\nauthorityKeyIdentifier = keyid:always\n";
    }
    // Quote paths to tolerate spaces; silence openssl's stderr chatter.
    std::string cmd = "openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 3650"
                      " -keyout \"" +
                      key + "\" -out \"" + cert + "\" -config \"" + cnf + "\" >/dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    fs::remove(cnf, ec);
    if (rc != 0 || !fs::exists(cert, ec) || !fs::exists(key, ec)) {
        fprintf(stderr, "crispasr: C2PA auto-cert generation failed (openssl missing?); "
                        "signing skipped (watermark still applied)\n");
        return false;
    }
    out_cert = cert;
    out_key = key;
    return true;
#endif // __EMSCRIPTEN__
}

// Print a one-time startup warning if C2PA is not compiled in.
inline void crispasr_c2pa_startup_check() {
#ifndef CRISPASR_HAVE_C2PA
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "crispasr: C2PA signing disabled (c2pa-c library not found; run "
                        "scripts/fetch-c2pa.sh and rebuild)\n");
        warned = true;
    }
#endif
}
