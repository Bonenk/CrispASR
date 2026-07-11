// crispasr_opus_writer.h — Ogg Opus serializer for TTS float32 PCM, built on
// the in-tree glint Opus encoder (glint_opus_encode_file). Header-only like
// crispasr_mp3_writer.h / crispasr_aac_writer.h; consumers link the `glint`
// static library.
//
// Unlike the MP3/AAC writers this does NOT prepend an ID3v2 AI-provenance tag:
// an Ogg stream must begin with the "OggS" capture pattern, so a leading ID3
// tag would corrupt it. (AI provenance belongs in the OpusTags comment header;
// exposing that through glint's C ABI is a follow-up.)
//
// glint's Opus encoder is CELT-only, fullband, 48 kHz — input is resampled to
// 48 kHz here (linear, good enough for speech; mirrors the server's Opus path).
// The result is a standard, playable .opus file (verified decodable by ffmpeg
// and libopus).

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <glint/glint.h>

// Encode float32 mono PCM in [-1, 1] to a complete Ogg Opus stream via glint.
// Samples outside [-1, 1] are clamped. Non-48 kHz input is linearly resampled
// to 48 kHz first (glint Opus is 48 kHz only). Returns empty on failure.
inline std::string crispasr_make_opus_glint(const float* pcm, int n_samples, int sample_rate, int bitrate_bps = 64000) {
    if (!pcm || n_samples <= 0 || sample_rate <= 0)
        return {};

    std::vector<float> resampled;
    if (sample_rate != 48000) {
        const int out_n = (int)((int64_t)n_samples * 48000 / sample_rate);
        if (out_n <= 0)
            return {};
        resampled.resize(out_n);
        for (int i = 0; i < out_n; i++) {
            float pos = (float)i * (float)sample_rate / 48000.0f;
            int s0 = (int)pos;
            int s1 = std::min(s0 + 1, n_samples - 1);
            float frac = pos - (float)s0;
            resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
        }
        pcm = resampled.data();
        n_samples = out_n;
    }

    // Clamp to [-1, 1] (same semantics as the MP3/AAC writers).
    std::vector<float> clamped(pcm, pcm + n_samples);
    for (float& s : clamped) {
        if (s > 1.0f)
            s = 1.0f;
        if (s < -1.0f)
            s = -1.0f;
    }

    int size = 0;
    uint8_t* data = glint_opus_encode_file(clamped.data(), n_samples, 1, bitrate_bps, /*vbr=*/0, &size);
    if (!data || size <= 0) {
        if (data)
            glint_free(data);
        return {};
    }
    std::string out((const char*)data, (size_t)size);
    glint_free(data);
    return out;
}

// Public entry: glint Ogg Opus. Kept as a thin wrapper so a future libopus A/B
// path can slot in behind CRISPASR_OPUS_ENCODER without touching call sites.
inline std::string crispasr_make_opus(const float* pcm, int n_samples, int sample_rate, int bitrate_bps = 64000) {
    return crispasr_make_opus_glint(pcm, n_samples, sample_rate, bitrate_bps);
}
