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
            for name in ('ignis_trellis_candidate.h','ignis_trellis_linear.h'):
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


class EngineProtocolTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
