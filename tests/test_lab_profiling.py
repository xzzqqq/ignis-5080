import sqlite3
import contextlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lab_support.profiling import summarize, profile, command
import subprocess


class TimelineTests(unittest.TestCase):
    def database(self, path, rows):
        with contextlib.closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
            connection.executemany('INSERT INTO StringIds VALUES (?, ?)', [(1, 'down'), (2, 'reduce')])
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER, deviceId INTEGER, contextId INTEGER, streamId INTEGER)')
            connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)', rows)

    def test_overlap_and_gaps_are_separate_from_summed_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'profile.sqlite'
            self.database(database, [(1000, 5000, 1, 0, 1, 1), (3000, 6000, 2, 0, 1, 2),
                                     (9000, 11000, 1, 0, 1, 1)])
            result = summarize(database)
        self.assertEqual(result['kernel_count'], 3)
        self.assertEqual(result['kernels'][0]['name'], 'down')
        self.assertEqual(result['kernels'][0]['count'], 2)
        self.assertEqual(result['kernels'][0]['total_us'], 6)
        self.assertEqual(result['devices'], [dict(device=0, kernel_span_us=10, kernel_busy_us=7,
                                                 kernel_gap_us=3, overlap_excess_us=2)])
        self.assertEqual(result['timeline'][1]['start_us'], 2)

    def test_devices_have_independent_time_unions(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'profile.sqlite'
            self.database(database, [(0, 2000, 1, 0, 1, 1), (0, 2000, 1, 1, 1, 1)])
            result = summarize(database)
        self.assertEqual(len(result['devices']), 2)
        self.assertTrue(all(device['overlap_excess_us'] == 0 for device in result['devices']))

    def test_empty_capture_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'profile.sqlite'
            self.database(database, [])
            with self.assertRaisesRegex(ValueError, 'no CUDA kernel executions'):
                summarize(database)

    def test_missing_cuda_table_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'profile.sqlite'
            with contextlib.closing(sqlite3.connect(database)):
                pass
            with self.assertRaisesRegex(ValueError, 'no CUDA kernel timeline'):
                summarize(database)


class CollectionTests(unittest.TestCase):
    def owner(self, fail=False):
        calls = []
        def call(payload):
            calls.append(payload)
            if fail and payload['op'] == 'replay_batch' and len(calls) > 1:
                raise RuntimeError('execution failed')
            return dict(ok=True, result=[] if payload['op'] == 'replay_batch' else {})
        record = dict(id=1, weight={'name':'layer.0.down'}, tokens=8)
        return SimpleNamespace(engine=SimpleNamespace(profiler_session='session', call=call),
            config=dict(warmup=1, samples=3, timeout_s=1), calls=calls,
            selection=[dict(data='real-inputs.bin', records=[record, dict(record, id=2)])])

    def test_capture_only_after_warmup_and_uses_representative_real_inputs(self):
        owner = self.owner()
        commands = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def run(argv, log, required=True, timeout=None):
                commands.append(argv)
                if argv[1] == 'stop':
                    (directory/'operator-profile.nsys-rep').touch()
                return 0
            with patch('lab_support.profiling.shutil.which', return_value='/bin/nsys'), \
                    patch('lab_support.profiling.command', side_effect=run), \
                    patch('lab_support.profiling.summarize', return_value=dict(kernel_count=2, timeline=[])):
                result = profile(owner, directory, {}, dict(identity='candidate'))
        self.assertTrue(result['passed'])
        self.assertEqual(result['representatives'], 1)
        self.assertEqual([call['op'] for call in owner.calls],
                         ['replay_batch', 'profiler_start', 'replay_batch', 'profiler_stop'])
        self.assertEqual(owner.calls[2]['data'], 'real-inputs.bin')
        self.assertEqual(owner.calls[2]['samples'], 3)
        self.assertTrue(owner.calls[2]['exploratory'])
        self.assertIn('--capture-range=cudaProfilerApi', commands[0])
        self.assertEqual(commands[-1][1], 'export')

    def test_replay_failure_closes_native_range_and_collection(self):
        owner = self.owner(fail=True)
        with tempfile.TemporaryDirectory() as temporary, \
                patch('lab_support.profiling.shutil.which', return_value='/bin/nsys'), \
                patch('lab_support.profiling.command', return_value=0) as run:
            with self.assertRaisesRegex(RuntimeError, 'execution failed'):
                profile(owner, Path(temporary), {}, dict(identity='candidate'))
        self.assertEqual(owner.calls[-1]['op'], 'profiler_stop')
        self.assertEqual(run.call_args.args[0][1], 'stop')

    def test_uninstrumented_session_reports_prerequisite(self):
        owner = self.owner()
        owner.engine.profiler_session = None
        with self.assertRaisesRegex(ValueError, 'requires nsys'):
            profile(owner, Path('/unused'), {}, dict(identity='candidate'))
        self.assertEqual(owner.calls, [])

    def test_cleanup_subprocess_timeout_requires_session_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch('lab_support.profiling.subprocess.run', side_effect=subprocess.TimeoutExpired('nsys', 10)):
            with self.assertRaisesRegex(RuntimeError, 'Nsight command timed out'):
                command(['nsys','stop'], Path(temporary)/'stop.log', required=False, timeout=10)


if __name__ == '__main__':
    unittest.main()
