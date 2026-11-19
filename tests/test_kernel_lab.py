"""CPU regression tests for experiment admission and tier short-circuiting."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
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
            for name in ('ignis_trellis_candidate.h','ignis_trellis_linear.h','ignis_gqa_candidate.h','ignis_segment_candidate.h','ignis_mlp_segment.h','ignis_gqa_segment.h','ignis_gdn_segment.h'):
                (source/'kernel/include'/name).write_text('abi')
            shared=source/'kernel/src/shared.cuh';shared.write_text('shared')
            def deps(src,name,*args):
                p=src/'kernel/src'/name
                result={str(p):builder.digest(p)}
                if name in builder.SOURCES[:2]:result[str(shared)]=builder.digest(shared)
                return result
            def compile(argv, **kwargs):
                Path(argv[-1]).write_bytes(b'compiled binary')
                return type('Exit', (), {'returncode': 0})()
            with patch.object(builder,'dependencies',side_effect=deps), patch.object(builder.subprocess, 'check_output', return_value='nvcc test'), patch.object(builder.subprocess, 'run', side_effect=compile) as run:
                first = builder.build(source, base / 'out', 'nvcc')
                same = builder.build(source, base / 'out', 'nvcc')
                self.assertTrue(same['cache_hit'])
                self.assertEqual(first['path'], same['path'])
                self.assertEqual(run.call_count, len(builder.SOURCES)+1)
                (source / 'kernel/src' / builder.SOURCES[0]).write_text('new candidate')
                second = builder.build(source, base / 'out', 'nvcc')
                self.assertNotEqual(first['path'], second['path'])
                self.assertEqual(run.call_count,len(builder.SOURCES)+3)
                self.assertEqual(sum(not o['cache_hit'] for o in second['objects']),1)
                self.assertEqual(Path(first['path']).read_bytes(), b'compiled binary')
                (source/'kernel/src/unrelated.cuh').write_text('unrelated')
                self.assertTrue(builder.build(source,base/'out','nvcc')['cache_hit'])
                shared.write_text('changed dependency')
                third=builder.build(source,base/'out','nvcc')
                self.assertEqual(sum(not o['cache_hit'] for o in third['objects']),2)
                Path(third['path']).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    builder.build(source, base / 'out', 'nvcc')


    def test_freeze_uses_exact_transitive_local_dependencies(self):
        import tempfile
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('candidate_builder_freeze',ROOT/'scripts/build_trellis_candidate.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as t:
            source=(Path(t)/'source').resolve();(source/'kernel/src').mkdir(parents=True)
            nested=source/'kernel/vendor/src/ops/kernel/hq_codec.cuh'
            nested.parent.mkdir(parents=True);nested.write_text('codec')
            unrelated=source/'kernel/vendor/src/ops/kernel/unrelated.cuh';unrelated.write_text('other')
            outside=Path(t)/'toolkit.h';outside.write_text('external')
            deps={str(nested):'codec',str(outside):'external'}
            with patch.object(builder,'dependencies',return_value=deps):
                files=builder.freeze_inputs(source,'nvcc')
            self.assertIn(nested,files);self.assertNotIn(unrelated,files);self.assertNotIn(outside,files)
            self.assertIn(source/'kernel/src/lab_gqa_candidate.cu',files)
            self.assertIn(source/'kernel/include/ignis_gqa_candidate.h',files)

    def test_gqa_hot_source_admission_is_explicit(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from lab_support.common import host_changes
        supported=['kernel/src/lab_gqa_candidate.cu','kernel/src/lab_gqa_candidate_policy.cuh',
            'kernel/src/gqa_hq_decode_impl.cuh','kernel/src/gqa_hq_decode_policy.cuh',
            'kernel/src/gqa_token_mixer_impl.cuh',
            'kernel/include/ignis_gqa_hq_reduce.cuh','kernel/vendor/src/ops/kernel/hq_codec.cuh']
        for path in supported:self.assertEqual(host_changes({path:'before'},{path:'after'}),[])
        for path in ['kernel/include/ignis_gqa_candidate.h','kernel/vendor/src/core/tensor.h',
                     'kernel/include/ignis_gqa_native.h','kernel/src/gqa_decode_hq.cu',
                     'kernel/src/gqa_native_segment.cu',
                     'kernel/vendor/manifest.json','kernel/src/gqa_candidate.h']:
            self.assertEqual(host_changes({path:'before'},{path:'after'}),[path])


class EngineProtocolTests(unittest.TestCase):
    def test_plain_session_ignores_installed_nsight_and_uses_direct_pipes(self):
        import os
        import sys
        import tempfile
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'scripts'))
        from kernel_lab import Session
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);marker=base/'nsys-invoked';launcher=base/'nsys'
            launcher.write_text('#!'+sys.executable+'\nfrom pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise SystemExit(9)\n')
            launcher.chmod(0o755)
            engine=base/'engine.py'
            engine.write_text("import sys,json,os\nprint(json.dumps({'event':'ready','pid':os.getpid()}),flush=True)\n"
                "for line in sys.stdin:\n request=json.loads(line)\n print(json.dumps({'id':request['id'],'ok':True}),flush=True)\n")
            with patch.dict(os.environ,PATH=str(base)+os.pathsep+os.environ.get('PATH','')):
                session=Session(Path(sys.executable),engine,base/'unused',base,1)
                try:
                    self.assertTrue(session.call({'op':'ping'})['ok'])
                    self.assertEqual(session.engine_pid,session.process.pid)
                    self.assertIsNone(session.transport_path)
                    self.assertIs(session.output_stream,session.process.stdout)
                    self.assertFalse((base/'engine.stdout.fifo').exists())
                    self.assertFalse(marker.exists())
                finally:session.close()

    def test_malformed_or_stale_reply_cannot_be_a_reload_ack(self):
        import tempfile
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from kernel_lab import Session
        for reply in ["print('debug output',flush=True)", "print(json.dumps({'ok':True,'id':request['id']-1}),flush=True)"]:
            with self.subTest(reply=reply),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);script=base/'engine.py'
                script.write_text("import sys,json\nprint(json.dumps({'event':'ready'}),flush=True)\nfor line in sys.stdin:\n request=json.loads(line)\n "+reply+"\n")
                session=Session(Path(sys.executable),script,base/'unused',base,5)
                try:
                    with self.assertRaises(RuntimeError):session.call({'op':'reload','path':'builtin'})
                finally:session.close()
    def profiled_session(self, base, body):
        import os
        import sys
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'scripts'))
        from kernel_lab import Session
        launcher=base/'nsys'
        launcher.write_text('#!'+sys.executable+'\n'+
            "import sys,subprocess,time\n"
            "print('Capture range started in the application.',flush=True)\n"
            "print(\"Generating '/tmp/nsys/report.qdstrm'\",flush=True)\n"
            "child=subprocess.Popen(sys.argv[sys.argv.index('/bin/bash'):])\n"
            "while child.poll() is None:\n print(' [1/1] arbitrary profiler progress...Processing events...',flush=True)\n time.sleep(.005)\n"
            "print('Capture range ended in the application.',flush=True)\nsys.exit(child.returncode)\n")
        launcher.chmod(0o755)
        engine=base/'engine.py'
        engine.write_text("import sys,json,os,time\nprint(json.dumps({'event':'ready','pid':os.getpid()}),flush=True)\n"
            "for line in sys.stdin:\n "+body+"\n")
        with patch.dict(os.environ,PATH=str(base)+os.pathsep+os.environ.get('PATH','')):
            return Session(Path(sys.executable),engine,base/'unused',base,5,profiler_session='test')

    def test_profiled_fifo_separates_arbitrary_launcher_progress(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            session=self.profiled_session(base,"request=json.loads(line); print(json.dumps({'ok':True,'id':request['id']}),flush=True)")
            fifo=base/'engine.stdout.fifo'
            try:
                self.assertTrue(fifo.exists())
                self.assertNotEqual(session.engine_pid,session.process.pid)
                self.assertTrue(session.call({'op':'profiler_stop'})['ok'])
                self.assertTrue(session.call({'op':'reload','path':'builtin'})['ok'])
            finally:session.close()
            self.assertFalse(fifo.exists())
            logs=(base/'engine.profiler.log').read_text()
            self.assertIn("Generating '/tmp/nsys/report.qdstrm'",logs)
            self.assertIn('arbitrary profiler progress',logs)
            self.assertIn('Capture range ended in the application.',logs)
            replies=[json.loads(line) for line in (base/'engine.jsonl').read_text().splitlines()]
            self.assertEqual([r.get('id') for r in replies],[None,1,2])

    def test_profiled_fifo_rejects_every_malformed_engine_reply(self):
        import tempfile
        for line in ('debug output','Capture range started in the application.',
                     'Capture range ended in the application. unexpected'):
            with self.subTest(line=line),tempfile.TemporaryDirectory() as directory:
                session=self.profiled_session(Path(directory),'print('+repr(line)+',flush=True)')
                try:
                    with self.assertRaisesRegex(RuntimeError,'invalid engine JSON'):
                        session.call({'op':'profiler_stop'})
                finally:session.close()

    def test_launcher_progress_does_not_extend_engine_deadline(self):
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as directory:
            session=self.profiled_session(Path(directory),'time.sleep(.2)')
            session.timeout=.04;began=time.monotonic()
            try:
                with self.assertRaises(TimeoutError):session.call({'op':'profiler_stop'})
                self.assertLess(time.monotonic()-began,.15)
            finally:session.close()

    def test_profiled_fifo_detects_exited_launcher_without_eof(self):
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as directory:
            session=self.profiled_session(Path(directory),'sys.exit(0)')
            session.timeout=1;began=time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError,'engine exited'):
                    session.call({'op':'profiler_stop'})
                self.assertLess(time.monotonic()-began,.5)
            finally:session.close()


if __name__ == '__main__':
    unittest.main()
