"""Non-vacuous strict model proof/parser and fixture extraction controls."""
import importlib.util
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

spec = importlib.util.spec_from_file_location('validator', Path(__file__).resolve().parents[1]/'tools/validate_cann_model.py')
validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)

CPU = '''backend_validation[moonshine.encoder]: assigned=CPU/MUL_MAT nodes=39
backend_validation[moonshine.encoder]: required_matmuls=39 unassigned=0 compute_success=1
'''
CANN = '''backend_validation: registry=CANN device=CANN1 ordinal=1 initialized=1
backend_validation[moonshine.encoder]: assigned=CANN1/MUL_MAT nodes=20
backend_validation[moonshine.encoder]: assigned=CPU/MUL_MAT nodes=19
backend_validation[moonshine.encoder]: required_matmuls=20 unassigned=0 compute_success=1
'''


class ModelValidatorTest(unittest.TestCase):
    def test_real_cpu_log_contract(self):
        result = validator.assess_placement(CPU*2, 'cpu', 0, 2)
        self.assertEqual(result['encoder_computes'], 2)

    def test_cann_permits_visible_cpu_fallback_but_requires_selected_npu(self):
        result = validator.assess_placement(CANN, 'cann', 1, 1)
        self.assertEqual(result['assigned_devices'], ['CANN1'])
        with self.assertRaises(ValueError): validator.assess_placement(CPU, 'cann', 0, 1)

    def test_missing_or_wrong_device_is_rejected(self):
        for text in (CANN.split('\n',1)[1], CANN.replace('ordinal=1','ordinal=0'),
                     CANN.replace('assigned=CANN1','assigned=CANN0'), CANN.replace('initialized=1','initialized=0')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validator.assess_placement(text, 'cann', 1, 1)

    def test_noop_partial_and_failed_encoder_cannot_pass(self):
        for text in ('', CPU.replace('required_matmuls=39','required_matmuls=0'),
                     CPU.replace('unassigned=0','unassigned=3'), CPU.replace('compute_success=1','compute_success=0'), CPU):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validator.assess_placement(text, 'cpu', 0, 2)

    def test_nz_on_requires_actual_model_selection(self):
        evidence = 'backend_validation[moonshine.encoder]: nz_cached_matmuls=20 nz_unique_weights=20 nz_bytes=8192 nz_selections=20 nz_selected_weights=20\n'
        result = validator.assess_placement(CANN+evidence, 'cann', 1, 1, 'on')
        self.assertEqual(result['nz_usage'][0]['selected_weights'], 20)
        for text in (CANN, CANN+evidence.replace('nz_selected_weights=20', 'nz_selected_weights=0'),
                     CANN+evidence.replace('nz_selections=20', 'nz_selections=0'), CANN+evidence*2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validator.assess_placement(text, 'cann', 1, 1, 'on')
        with self.assertRaises(ValueError):
            validator.assess_placement(CANN+evidence, 'cann', 1, 1, 'off')
        empty = 'backend_validation[moonshine.encoder]: nz_cached_matmuls=0 nz_unique_weights=0 nz_bytes=0 nz_selections=0 nz_selected_weights=0\n'
        validator.assess_placement(CANN+empty, 'cann', 1, 1, 'off')
        with self.assertRaises(ValueError):
            validator.assess_placement(CANN+empty, 'cann', 1, 1, 'on')

    def test_metal_requires_exact_device_assignment(self):
        text = CANN.replace('CANN', 'MTL').replace('registry=MTL', 'registry=Metal')
        result = validator.assess_placement(text, 'metal', 1, 1)
        self.assertEqual(result['assigned_devices'], ['MTL1'])
        with self.assertRaises(ValueError):
            validator.assess_placement(CPU, 'metal', 0, 1)

    def test_gpu_flags_cannot_be_silently_ignored(self):
        helper=Path(validator.fixture.__file__)
        for args in (['--use-gpu'], ['--encoder-only'], ['--native']):
            process=subprocess.run([sys.executable,str(helper)]+args,capture_output=True,text=True)
            self.assertEqual(process.returncode,2)
            self.assertIn('requires' if args==['--native'] else 'require --native',process.stderr)

    def test_safe_archive_extracts_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'fixture.zip';out=root/'output'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('nested/reference.json',b'{"original":true}')
            validator.unpack(archive,out)
            self.assertEqual((out/'nested/reference.json').read_bytes(),b'{"original":true}')

    def test_traversal_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'fixture.zip';out=root/'output'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('ordinary.txt','safe');z.writestr('../escaped.txt','unsafe')
            with self.assertRaises(ValueError):validator.unpack(archive,out)
            self.assertFalse(out.exists());self.assertFalse((root/'escaped.txt').exists())

    def test_duplicate_absolute_and_symlink_members_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('duplicate','absolute','symlink'):
                archive=root/(name+'.zip')
                with zipfile.ZipFile(archive,'w') as z:
                    if name=='duplicate':
                        z.writestr('a','first')
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore');z.writestr('a','second')
                    elif name=='absolute':z.writestr('/escaped','unsafe')
                    else:
                        info=zipfile.ZipInfo('link');info.create_system=3
                        info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,'../target')
                with self.subTest(name=name),self.assertRaises(ValueError):validator.unpack(archive,root/name)


if __name__=='__main__':unittest.main()
