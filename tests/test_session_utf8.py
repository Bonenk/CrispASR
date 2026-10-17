"""Compile production session finalization/getters; exercise strict ctypes UTF-8.

No model mock purports to test decoder quality: the injected decoded bytes
reproduce the ABI failure independently of weights and GPU availability.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def definition(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


class SessionUtf8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'src/crispasr_c_api.cpp').read_text()
        cls.tmp = tempfile.TemporaryDirectory()
        code = '#include <string>\n#include <vector>\n#include <cstdint>\n#include "core/utf8.h"\n#define CA_EXPORT extern "C"\n'
        for marker in ('struct crispasr_session_seg {', 'struct crispasr_session_result {'):
            code += definition(source, marker) + ';\n'
        marker = 'static void normalize_session_result_utf8('
        if marker in source:
            code += definition(source, marker) + '\n'
        code += r'''
struct crispasr_session { int best_of = 1; std::string backend = "qwen3-asr"; bool hygiene_defer_merge = false; };
static std::string injected;
static std::string callback_text;
static bool session_language_satisfiable(crispasr_session*, const char*) { return true; }
static crispasr_session_result* transcribe_autochunk(crispasr_session*, const float*, int, const char*) {
    auto* r = new crispasr_session_result;
    crispasr_session_seg seg;
    seg.text = injected; seg.t0 = 123; seg.t1 = 456;
    crispasr_session_seg::word w;
    w.text = injected; w.t0 = 130; w.t1 = 450; w.p = .75f;
    w.alts.push_back({injected, .5f});
    seg.words.push_back(w); r->segments.push_back(seg);
    return r;
}
static void apply_session_hygiene(crispasr_session_result*, bool) {}
static void apply_session_punc_model(crispasr_session*, crispasr_session_result*) {}
static void _fire_segment_callbacks(crispasr_session*, crispasr_session_result* r) {
    callback_text = r ? r->segments[0].text : "";
}
'''
        for name in ('transcribe_lang', 'transcribe', 'result_segment_text', 'result_word_text',
                     'result_segment_t0', 'result_segment_t1', 'result_word_t0', 'result_word_t1', 'result_word_p'):
            pos = source.index(' crispasr_session_' + name + '(')
            start = source.rfind('CA_EXPORT ', 0, pos)
            code += definition(source, source[start:pos + len(' crispasr_session_' + name + '(')]) + '\n'
        code += r'''
extern "C" crispasr_session_result* inject(const char* text, int n, int best_of, bool legacy) {
    injected.assign(text, n); crispasr_session s; s.best_of = best_of; float pcm = 0;
    return legacy ? crispasr_session_transcribe(&s, &pcm, 1)
                  : crispasr_session_transcribe_lang(&s, &pcm, 1, "zh");
}
extern "C" const char* callback() { return callback_text.c_str(); }
extern "C" const char* alternative(crispasr_session_result* r) { return r->segments[0].words[0].alts[0].text.c_str(); }
extern "C" void dispose(crispasr_session_result* r) { delete r; }
'''
        path = Path(cls.tmp.name)
        (path / 'test.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-shared', '-fPIC',
                        '-I', str(ROOT / 'src'), str(path / 'test.cpp'), '-o', str(path / 'test.so')], check=True)
        cls.lib = ctypes.CDLL(str(path / 'test.so'))
        cls.lib.inject.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        cls.lib.inject.restype = ctypes.c_void_p
        cls.lib.dispose.argtypes = [ctypes.c_void_p]
        cls.lib.callback.restype = ctypes.c_char_p
        cls.lib.alternative.argtypes = [ctypes.c_void_p]
        cls.lib.alternative.restype = ctypes.c_char_p
        for name, count, restype in [('segment_text', 1, ctypes.c_char_p), ('word_text', 2, ctypes.c_char_p),
                                     ('segment_t0', 1, ctypes.c_int64), ('segment_t1', 1, ctypes.c_int64),
                                     ('word_t0', 2, ctypes.c_int64), ('word_t1', 2, ctypes.c_int64),
                                     ('word_p', 2, ctypes.c_float)]:
            fn = getattr(cls.lib, 'crispasr_session_result_' + name)
            fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * count
            fn.restype = restype

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_text(self, raw, expected):
        for best_of in (1, 3):
            for legacy in (False, True):
                with self.subTest(raw=raw, best_of=best_of, legacy=legacy):
                    r = self.lib.inject(raw, len(raw), best_of, legacy)
                    try:
                        for value in (self.lib.crispasr_session_result_segment_text(r, 0),
                                      self.lib.crispasr_session_result_word_text(r, 0, 0),
                                      self.lib.alternative(r), self.lib.callback()):
                            self.assertEqual(value.decode('utf-8'), expected)
                        self.assertEqual(self.lib.crispasr_session_result_segment_t0(r, 0), 123)
                        self.assertEqual(self.lib.crispasr_session_result_segment_t1(r, 0), 456)
                        self.assertEqual(self.lib.crispasr_session_result_word_t0(r, 0, 0), 130)
                        self.assertEqual(self.lib.crispasr_session_result_word_t1(r, 0, 0), 450)
                        self.assertEqual(self.lib.crispasr_session_result_word_p(r, 0, 0), .75)
                        self.assertEqual(self.lib.crispasr_session_result_word_text(r, -1, 0), b'')
                        self.assertEqual(self.lib.crispasr_session_result_segment_text(None, 0), b'')
                    finally:
                        self.lib.dispose(r)

    def test_valid_multilingual_and_assembled_byte_tokens(self):
        text = '中文 하, 둘, 셋. 日本語 café 😀'
        self.check_text(text.encode(), text)
        # Same Korean split-byte BPE sequence as issue #475: normalize AFTER joining.
        pieces = [b'\xed\x95\x98', b',', b' \xeb', b'\x91', b'\x98', b',', b' \xec', b'\x85', b'\x8b', b'.']
        self.check_text(b''.join(pieces), '하, 둘, 셋.')

    def test_stray_continuation_keeps_rest_of_chunk(self):
        self.check_text('前'.encode() + b'\x84' + '后'.encode(), '前�后')

    def test_truncated_final_character(self):
        self.check_text('完整'.encode() + b'\xe4\xb8', '完整��')

    def test_rfc3629_invalid_sequences(self):
        for raw in (b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xf5\x80\x80\x80'):
            self.check_text(b'A' + raw + b'Z', 'A' + '�' * len(raw) + 'Z')


if __name__ == '__main__':
    unittest.main()
