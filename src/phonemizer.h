// phonemizer.h — pluggable text-to-phoneme interface.
//
// Provides a common abstraction for phonemization backends:
//   1. espeak-ng (via dlopen or popen) — GPLv3, loaded at runtime
//   2. [future] CMUdict lookup — public domain, English-only
//   3. [future] Neural G2P — MIT/Apache, multilingual
//   4. [future] GGUF-embedded dictionary — zero dependencies
//
// Each backend implements the same interface. The runtime tries them
// in priority order until one succeeds.

#pragma once

#include <string>
#include <vector>
#include <functional>

namespace crispasr {

// Phonemizer backend interface.
// text  = UTF-8 input text (e.g. "Hello world")
// lang  = espeak-ng voice name or BCP-47 tag (e.g. "en-us")
// out   = IPA phoneme string (e.g. "həlˈoʊ wˈɜːld")
// Returns true on success.
using phonemize_fn = std::function<bool(const std::string& lang,
                                         const std::string& text,
                                         std::string& out)>;

// Built-in backend: espeak-ng via dlopen (MIT-clean, loads GPL at runtime).
// Returns false if libespeak-ng is not available.
bool phonemize_espeak_dlopen(const std::string& lang, const std::string& text, std::string& out);

// Built-in backend: espeak-ng via popen subprocess.
// Returns false if the espeak-ng binary is not on $PATH.
bool phonemize_espeak_popen(const std::string& lang, const std::string& text, std::string& out);

// Built-in English G2P: LTS rules (always available, zero deps) +
// optional CMUdict (134K words, auto-loaded from ~/.cache/crispasr/cmudict.dict
// or CRISPASR_CMUDICT_PATH env var) + optional neural G2P (GRU seq2seq).
// Produces IPA directly via ARPAbet→IPA conversion table.
// For non-English, returns false and falls through.
bool phonemize_builtin_en(const std::string& lang, const std::string& text, std::string& out);

// Built-in German G2P: LTS rules (always available) + optional IPA
// dictionary (787K words, auto-loaded from ~/.cache/crispasr/ipa_dict_de.txt
// or CRISPASR_DE_DICT_PATH env var, CC-BY-SA from open-dict-data).
// For non-German, returns false and falls through.
bool phonemize_builtin_de(const std::string& lang, const std::string& text, std::string& out);

// Try all available phonemizers in priority order.
// Order: builtin_en → builtin_de → espeak_dlopen → espeak_popen
inline bool phonemize(const std::string& lang, const std::string& text, std::string& out) {
    if (phonemize_builtin_en(lang, text, out)) return true;
    if (phonemize_builtin_de(lang, text, out)) return true;
    if (phonemize_espeak_dlopen(lang, text, out)) return true;
    if (phonemize_espeak_popen(lang, text, out)) return true;
    return false;
}

} // namespace crispasr
