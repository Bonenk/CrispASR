"""Real opaque Whisper parameter adapters with a model-free decoder."""
import ctypes as C
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
import numpy as np
from build_legacy_wrapper_probe import build
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from crispasr import CrispASR, Parakeet, TitaNet

class LegacyPointerContract(unittest.TestCase):
    def test_native_parameters_parallel_dtw_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='python-legacy-') as tmp:
            path=build(Path(tmp)/'libcrispasr.so');lib=C.CDLL(str(path))
            with self.assertRaises(ValueError):CrispASR('probe',lib_path=str(path),dtw=True,dtw_aheads_preset=2)
            with self.assertRaises(RuntimeError):CrispASR('fail',lib_path=str(path))
            self.assertEqual(lib.probe_legacy_live(),0)
            model=CrispASR('probe',lib_path=str(path),dtw=True,dtw_n_top=2)
            self.assertEqual(lib.probe_legacy_dtw(),102)
            self.assertEqual(lib.probe_legacy_live(),0)
            # Exercise native pointer inference even on clones with a built helper.
            model._helpers = None
            pcm=np.array([1,3,5],dtype=np.float32)
            result=model.transcribe_pcm(pcm,language='de',n_processors=2,decode_options={'initial_prompt':'Grüße','token_timestamps':True})
            self.assertEqual(lib.probe_legacy_processors(),2);self.assertEqual(lib.probe_legacy_progress_zero(),1)
            self.assertEqual(result[0].text,'日');self.assertEqual(result[0].start,1)
            self.assertEqual(b''.join(t.raw_text for t in result[0].tokens),'日'.encode())
            self.assertIsNone(result[0].tokens[0].start)
            self.assertEqual(model.token_dtw_time(0,0),1.4)
            self.assertEqual(result[0].tokens[0].alternatives[0].token_id,42)
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(2)
            with self.assertRaises(RuntimeError):model.transcribe_pcm(pcm,language='de')
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(3)
            with self.assertRaises(RuntimeError):model.transcribe_pcm(pcm,language='de')
            self.assertEqual(lib.probe_legacy_live(),0)
            lib.probe_legacy_mode(0)
            with self.assertRaises(ValueError):model.transcribe_pcm(pcm,n_processors=0)
            self.assertEqual(lib.probe_legacy_live(),0)
            model.close();self.assertEqual(lib.probe_legacy_contexts(),0)

    def test_parakeet_uses_int64_offset_and_copies_partial_tokens(self):
        with tempfile.TemporaryDirectory(prefix='python-parakeet-') as tmp:
            path=Path(tmp)/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-parakeet-legacy.c'),'-o',str(path)],check=True)
            lib=C.CDLL(str(path));model=Parakeet('probe',lib_path=str(path));pcm=np.array([1,3,5],dtype=np.float32)
            result=model.transcribe(pcm,t_offset_cs=123)
            self.assertEqual(result['text'],'日本語');self.assertEqual(result['words'][0][1],223)
            self.assertEqual(result['tokens'][0][1],-1)
            self.assertEqual(b''.join(result['token_bytes']),'日'.encode());self.assertEqual(lib.probe_parakeet_results(),0)
            with self.assertRaises(ValueError):model.transcribe(pcm,language='de')
            with self.assertRaises(ValueError):model.transcribe(np.array([],dtype=np.float32))
            model.close();model.close();self.assertEqual(lib.probe_parakeet_contexts(),0)
            with self.assertRaises(RuntimeError):model.transcribe(pcm)

    def test_titanet_requires_192_float_success_and_rejects_closed_context(self):
        with tempfile.TemporaryDirectory(prefix='python-titanet-') as tmp:
            path=Path(tmp)/'libcrispasr.so'
            subprocess.run(['cc','-shared','-fPIC','-Werror','-I'+str(ROOT/'include'),str(ROOT/'tests/fixtures/wrapper-titanet-legacy.c'),'-o',str(path)],check=True)
            lib=C.CDLL(str(path));model=TitaNet('probe',lib_path=str(path));pcm=np.array([1,3,5],dtype=np.float32)
            result=model.embed(pcm);self.assertEqual(result.shape,(192,));self.assertEqual(result[0],1)
            for mode in (2,3,4):
                lib.probe_titanet_mode(mode)
                with self.assertRaises(RuntimeError):model.embed(pcm)
            lib.probe_titanet_mode(0)
            with self.assertRaises(ValueError):model.embed([])
            model.close();model.close();self.assertEqual(lib.probe_titanet_live(),0)
            calls=lib.probe_titanet_calls()
            with self.assertRaises(RuntimeError):model.embed(pcm)
            self.assertEqual(lib.probe_titanet_calls(),calls)

if __name__=='__main__':unittest.main()
