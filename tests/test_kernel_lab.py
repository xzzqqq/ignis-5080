"""CPU regression tests for experiment admission and tier short-circuiting."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kernel_lab', ROOT / 'scripts/kernel_lab.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class KernelLabTests(unittest.TestCase):
    def test_order_and_op_are_enforced(self):
        layer = dict(id='gdn', tier='layer', command={'op': 'layer'})
        forward = dict(id='decode', tier='forward', command={'op': 'measure'})
        lab.validate_plan({'cases': [layer, forward]})
        for cases in [[], [forward, layer], [layer, layer], [dict(layer, command={'op': 'reload'})]]:
            with self.assertRaises(ValueError):
                lab.validate_plan({'cases': cases})

    def test_tokens_alone_do_not_pass_numerical_gate(self):
        reference = dict(input_tokens=[12], logits_sha256=['golden'], tokens=[[1, 2]])
        self.assertTrue(lab.agrees(reference, dict(reference), 'forward'))
        self.assertFalse(lab.agrees(reference, dict(reference, logits_sha256=['wrong']), 'forward'))
        self.assertFalse(lab.agrees({}, {}, 'forward'))

    def test_warmup_failure_stops_before_timed_samples(self):
        class Fake:
            calls = 0
            def call(self, command):
                self.calls += 1
                return {'ok': True, 'result': dict(input_sha256='same', output_sha256='wrong')}
        session = Fake()
        result = lab.gate(session, dict(tier='layer', command={'op': 'layer'}),
                          dict(input_sha256='same', output_sha256='golden'), 1, 3)
        self.assertFalse(result['passed'])
        self.assertEqual(session.calls, 1)
        self.assertNotIn('median_ms', result)

    def test_engine_failure_is_not_a_timing(self):
        class Fake:
            def call(self, command):
                return {'ok': False, 'error': 'CUDA failure'}
        with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
            lab.sample(Fake(), {'command': {'op': 'measure'}}, 1)

class CandidateBuildTests(unittest.TestCase):
    def test_immutable_cache_checks_binary_and_tracks_changed_source(self):
        import tempfile
        from unittest.mock import patch
        builder_spec = importlib.util.spec_from_file_location('candidate_builder', ROOT / 'scripts/build_trellis_candidate.py')
        builder = importlib.util.module_from_spec(builder_spec)
        builder_spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'source'
            (source / 'kernel/src').mkdir(parents=True)
            (source / 'kernel/include').mkdir()
            for name in builder.SOURCES:
                (source / 'kernel/src' / name).write_text('original')
            def compile(argv, **kwargs):
                Path(argv[-1]).write_bytes(b'compiled binary')
                return type('Exit', (), {'returncode': 0})()
            with patch.object(builder.subprocess, 'check_output', return_value='nvcc test'), patch.object(builder.subprocess, 'run', side_effect=compile) as run:
                first = builder.build(source, base / 'out', 'nvcc')
                same = builder.build(source, base / 'out', 'nvcc')
                self.assertTrue(same['cache_hit'])
                self.assertEqual(first['path'], same['path'])
                self.assertEqual(run.call_count, 1)
                (source / 'kernel/src' / builder.SOURCES[0]).write_text('new candidate')
                second = builder.build(source, base / 'out', 'nvcc')
                self.assertNotEqual(first['path'], second['path'])
                self.assertEqual(Path(first['path']).read_bytes(), b'compiled binary')
                Path(second['path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    builder.build(source, base / 'out', 'nvcc')


if __name__ == '__main__':
    unittest.main()
