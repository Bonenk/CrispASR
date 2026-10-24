"""Model-free controls for selected-recipe Q4 publication; no network/model use."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
publisher = importlib.import_module('granite5_q4_publish')
from granite5_q4_recipe import RECIPES


class Q4PublicationGates(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'q4_k').mkdir()
        for label in ('stages', 'session', 'cli'):
            (self.root / 'q4_k' / (label + '.log')).write_text(
                'node # 1 (MUL_MAT): [CUDA0] use=1 output.weight\n')
        self.sources = {f'src/{i}.cpp': '0' * 64 for i in range(700)}
        file = dict(repo=publisher.STAGING, repo_type='model', revision='1' * 40, path='candidate.gguf',
                    sha256='2' * 64, bytes=100)
        validation = dict(accepted=True, complete=True, device='cpu', prompt_ids_exact=True,
                          bucketed={'exact': True}, dynamic={'exact': True})
        row = dict(cpu_acceptance=True, validation=validation, rules=RECIPES['body'], file=file,
                   elements_by_type={'Q4_K': 1024})
        self.cpu = dict(source='a' * 40, run=42, complete=True, diagnostic_only=True,
                        production_release=False, cuda_acceptance=False,
                        runtime_sources=self.sources, cases={'body': row})
        gpu_validation = dict(accepted=True, complete=True, device='cuda', prompt_ids_exact=True,
            bucketed={'exact': True}, dynamic={'exact': True}, executions={
            name: dict(returncode=0, peak_gpu_mib=1024, gpu_matmul_nodes=1, gpu_decoder_matmul_nodes=1)
            for name in ('stages', 'session', 'cli')})
        self.gpu = dict(complete=True, cuda_acceptance=True, production_release=False,
            selected_recipe='body', cpu_source=self.cpu['source'], cpu_run=42, cpu_receipt_sha256='b' * 64,
            runtime_source='c' * 40, runtime_provenance_sha256='d' * 64,
            profiles={'q4_k': dict(accepted=True, file=file, rules=RECIPES['body'], validation=gpu_validation)},
            hardware={'uuid': 'GPU-test'}, compiler_header_components={'CUB': dict(version='2.3.2',
                license='BSD-3-Clause', notice='licenses/CUB-2.3.2.txt',
                license_sha256='2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3')},
            rope_parity=dict(complete=True, device='cuda', cases=[dict(mode=i, backend='CUDA0',
                exit_code=0 if i == 1 else 1, maxabs=.001 if i == 1 else .1,
                low_position_maxabs=0., norm_ratio=1., relative_l2=0.) for i in range(5)]))
        self.preflight = dict(runtime_preflight_pass=True, gpu0={'uuid': 'GPU-test'})

    def qualify(self):
        return publisher.qualify(self.cpu, self.gpu, self.preflight, self.root,
            source='a' * 40, run=42, receipt_sha='b' * 64, case='body', actual_sources=self.sources)

    def test_valid_linked_evidence(self):
        self.assertEqual(self.qualify()['rules'], RECIPES['body'])

    def test_partial_stale_or_relabelled_approval_is_rejected(self):
        original_cpu, original_gpu = copy.deepcopy(self.cpu), copy.deepcopy(self.gpu)
        mutations = [lambda: self.cpu.update(complete=False),
            lambda: self.cpu['cases']['body'].update(cpu_acceptance=False),
            lambda: self.gpu.update(cpu_receipt_sha256='f' * 64),
            lambda: self.gpu.update(cpu_run=43),
            lambda: self.gpu['profiles']['q4_k'].update(file={}),
            lambda: self.gpu['profiles']['q4_k'].update(rules=[]),
            lambda: self.gpu['profiles']['q4_k']['validation'].update(prompt_ids_exact=False),
            lambda: self.gpu['profiles']['q4_k']['validation']['dynamic'].update(exact=False),
            lambda: self.cpu['cases']['body'].update(elements_by_type={'F16': 1024}),
            lambda: self.cpu.update(runtime_sources={}),
            lambda: self.gpu['compiler_header_components']['CUB'].update(license_sha256='0' * 64)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.cpu, self.gpu = copy.deepcopy(original_cpu), copy.deepcopy(original_gpu)
                mutate()
                with self.assertRaises(ValueError):
                    self.qualify()

    def test_each_process_requires_actual_decoder_nodes_and_residency(self):
        for label in ('stages', 'session', 'cli'):
            path = self.root / 'q4_k' / (label + '.log')
            original = path.read_text()
            path.write_text('node # 1 (MUL_MAT): [CPU] use=1 output.weight [CUDA0]\n')
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.qualify()
            path.write_text(original)
            self.gpu['profiles']['q4_k']['validation']['executions'][label]['peak_gpu_mib'] = 0
            with self.assertRaises(ValueError):
                self.qualify()
            self.gpu['profiles']['q4_k']['validation']['executions'][label]['peak_gpu_mib'] = 1024

    def test_wrong_rotary_control_and_device_are_rejected(self):
        self.gpu['rope_parity']['cases'][0]['maxabs'] = .001
        with self.assertRaises(ValueError):
            self.qualify()
        self.gpu['rope_parity']['cases'][0]['maxabs'] = .1
        self.preflight['gpu0']['uuid'] = 'wrong-device'
        with self.assertRaises(ValueError):
            self.qualify()

    def test_nested_evidence_preserves_parent_failures(self):
        parent = dict(parent_cuda_acceptance=False, parent_cuda_complete=False,
                      unaccepted_profiles=['q8_0', 'q4_k', 'q6_k'], profiles={'f16': {'sha256': 'original'}})
        frozen = copy.deepcopy(parent)
        result = publisher.extend_evidence(parent, {'cpu_acceptance': True})
        self.assertEqual(parent, frozen)
        self.assertEqual(result['parent_unaccepted_profiles'], frozen['unaccepted_profiles'])
        self.assertEqual(result['unaccepted_profiles'], ['q8_0', 'q6_k'])
        self.assertEqual(result['qualified_profiles'], ['f16', 'q4_k'])
        for key in ('parent_cuda_acceptance', 'parent_cuda_complete', 'profiles'):
            self.assertEqual(result[key], frozen[key])
        with self.assertRaises(ValueError):
            publisher.extend_evidence(result, {})

    def test_independent_numeric_and_exact_fidelity_are_mandatory(self):
        case = dict(checks_complete=True, official_request_text_exact=True,
                    original_vs_bucketed={'exact': True}, original_vs_dynamic={'exact': True})
        report = dict(structural_numerical_content_pass=True, cases={'q4_k': case})
        publisher.check_independent(report)
        for key in ('structural_numerical_content_pass', 'official_request_text_exact'):
            bad = copy.deepcopy(report)
            (bad if key.startswith('structural') else bad['cases']['q4_k'])[key] = False
            with self.assertRaises(ValueError):
                publisher.check_independent(bad)

    def test_mutable_revision_and_escaping_prefix_are_rejected(self):
        with self.assertRaises(ValueError):
            publisher.immutable('main', 40)
        for value in ('', '.', '../gpu', '/gpu', 'gpu/../../escape', 'gpu/*'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                publisher.prefix_path(value)


if __name__ == '__main__':
    unittest.main()
