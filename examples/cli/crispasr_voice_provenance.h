// crispasr_voice_provenance.h — the IO half of the voice-clone gate.
//
// Resolves a --voice / "voice" value to the file the backend will actually
// open, reads the provenance stamp out of a baked pack, and hands both to the
// pure predicate in crispasr_voice_clone_policy.h.
//
// Split from the policy header on purpose: the policy stays weight-free and
// IO-free so tests/test-voice-clone-policy.cpp can exercise it with no GGUF, no
// filesystem and no link against crispasr-core. Everything that needs a file
// lives here.

#pragma once

#include "core/gguf_loader.h"
#include "crispasr_voice_clone_policy.h"

#include <filesystem>
#include <string>

namespace crispasr_voice {

// True if `path` is a baked pack that declares it was derived from a real
// recording. False for presets, for unstamped packs, and for anything
// unreadable — an unreadable pack is not evidence of a clone, and the backend
// is about to fail on it anyway.
inline bool pack_declares_clone(const std::string& path) {
    if (path.empty() || !is_voice_pack(path))
        return false;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        return false;
    gguf_context* meta = core_gguf::open_metadata(path.c_str());
    if (!meta)
        return false;
    const bool cloned = core_gguf::kv_bool(meta, provenance_key(), false);
    core_gguf::free_metadata(meta);
    return cloned;
}

// Resolve the value a caller passed to the file a backend will open.
//
// An absolute/relative path is returned as-is. A BARE name (the server's
// documented form, e.g. {"voice": "victim"}) is resolved against `voice_dir`
// the way the TTS adapters resolve it — pack first, then recording. Resolving
// here is load-bearing, not cosmetic: the gate used to inspect the raw string,
// so a bare name reached the same .wav while scoring as "not a clone".
inline std::string resolve_voice_path(const std::string& voice, const std::string& voice_dir) {
    if (voice.empty() || voice_dir.empty())
        return voice;
    // Already a path or an explicit file — nothing to resolve.
    if (is_voice_pack(voice) || is_recording_reference(voice))
        return voice;
    if (voice.find('/') != std::string::npos || voice.find('\\') != std::string::npos)
        return voice;
    // Reject traversal before touching the filesystem (mirrors the adapters).
    if (voice.find("..") != std::string::npos || voice.find('\0') != std::string::npos)
        return voice;
    std::error_code ec;
    const std::string gguf = voice_dir + "/" + voice + ".gguf";
    if (std::filesystem::exists(gguf, ec))
        return gguf;
    const std::string wav = voice_dir + "/" + voice + ".wav";
    if (std::filesystem::exists(wav, ec))
        return wav;
    return voice;
}

// The whole gate in one call: resolve, read provenance, classify.
//
// `baked_from_wav_this_run` is the runtime's own knowledge that it baked this
// voice from a user recording during this run — it outranks anything the file
// says, and is the reason the TADA inline clone is gated at all.
inline CloneDecision classify_voice(const std::string& voice, const std::string& voice_dir,
                                    bool baked_from_wav_this_run) {
    const std::string resolved = resolve_voice_path(voice, voice_dir);
    // Skip the GGUF read when the answer can't depend on it — avoids opening a
    // file per request on the server's hot path for presets passed by name.
    const bool need_pack_read = !baked_from_wav_this_run && !is_recording_reference(resolved);
    const bool declares = need_pack_read && pack_declares_clone(resolved);
    return classify(resolved, baked_from_wav_this_run, declares);
}

} // namespace crispasr_voice
