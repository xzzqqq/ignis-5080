"""CPU contracts for Lab compiler storage, including existing native-cache links."""
import os
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import host, common
import build_trellis_candidate as builder


class LabBuildStorageTests(unittest.TestCase):
    def test_export_host_command_reads_and_dispatches_payload(self):
        payload = {'operation':'export-operation', 'config':{'data_dir':'/data'}}
        argv = ['host.py', '--root', '/repo', 'export', '--owner', 'owner']
        with patch.object(sys, 'argv', argv), patch.object(sys, 'stdin', io.StringIO(json.dumps(payload))), \
             patch.object(sys, 'stdout', io.StringIO()), patch.object(host.os, 'umask'), \
             patch.object(host, 'control', return_value={'passed':True}) as control:
            host.main()
        control.assert_called_once_with(Path('/repo'), 'owner', 'export', payload)

    def test_mount_lookup_resolves_links_and_escaped_nested_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            disk = root/'disk cache';disk.mkdir()
            link = root/'link';link.symlink_to(disk, target_is_directory=True)
            escaped = str(disk).replace(' ', r'\040')
            mounts = ('1 0 1:1 / / rw - ext4 /dev/root rw\n'
                      f'2 1 0:2 / {escaped} rw - tmpfs tmpfs rw\n')
            with patch.object(host.sys, 'platform', 'linux'), patch.object(Path, 'read_text', return_value=mounts):
                self.assertEqual(host.filesystem_type(link/'target'), 'tmpfs')
                self.assertEqual(host.filesystem_type(root/'unmounted'), 'ext4')

    def test_default_and_explicit_root_are_preserved(self):
        service = host.Owner(Path('/repo'), dict(config={'data_dir':'/data'}, directory='/session'))
        self.assertEqual(service.build_root, Path('/data/build'))
        service.config['build_root'] = '/explicit/cache'
        self.assertEqual(service.build_root, Path('/explicit/cache'))

    def test_build_environment_and_candidates_use_validated_disk(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory).resolve();(root/'kernel').mkdir()
            build = root/'disk'
            inherited=dict(os.environ)
            with patch.object(host, 'filesystem_type', return_value='ext4') as filesystem:
                env = host.prepare_build(root, build)
            self.assertEqual((root/'kernel/build').resolve(), build/'kernel')
            for name in ('target', 'kernel', 'tmp', 'candidates'):
                self.assertTrue((build/name).is_dir())
                self.assertIn(unittest.mock.call(build/name), filesystem.call_args_list)
            self.assertIn(unittest.mock.call(build/'candidates/objects'), filesystem.call_args_list)
            self.assertEqual(env['CARGO_TARGET_DIR'], str(build/'target'))
            self.assertEqual(env['TMPDIR'], str(build/'tmp'))
            self.assertEqual(dict(os.environ), inherited)

    def test_current_candidate_passes_disk_tmpdir_without_changing_owner_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, CUDA_HOME='/toolkit', TMPDIR='/inherited/tmp'):
            root=Path(directory).resolve()
            source=root/'kernel/src/candidate.cu';source.parent.mkdir(parents=True);source.write_text('source')
            library=root/'candidate.so';library.write_bytes(b'candidate')
            service=host.Owner(root,dict(config={'data_dir':str(root/'data')},directory=str(root)))
            service.evaluation_source={str(source.relative_to(root)):common.digest(source)}
            inherited=dict(os.environ)
            with patch.object(builder,'freeze_inputs',return_value={source}) as freeze, \
                 patch.object(host,'build_candidate',return_value=dict(path=str(library),identity='candidate')) as build:
                service.resolve_candidate('current',root/'operation')
            expected=dict(inherited,TMPDIR=str(service.build_root/'tmp'))
            self.assertEqual(freeze.call_args.kwargs['env'],expected)
            self.assertEqual(build.call_args.kwargs['env'],expected)
            self.assertEqual(dict(os.environ),inherited)

    def test_candidate_subprocesses_receive_explicit_environment_including_dependency_scan(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, TMPDIR='/inherited/tmp', CPATH='/inherited/include'):
            root=Path(directory).resolve();source=root/'source'
            code=source/'kernel/src/candidate.cu';code.parent.mkdir(parents=True);code.write_text('source')
            for name in ('ignis_trellis_candidate.h','ignis_trellis_linear.h','ignis_gqa_candidate.h','ignis_segment_candidate.h','ignis_mlp_segment.h','ignis_gqa_segment.h','ignis_gdn_segment.h'):
                header=source/'kernel/include'/name;header.parent.mkdir(exist_ok=True);header.write_text('abi')
            env=dict(os.environ,TMPDIR=str(root/'disk/tmp'),CPATH='/candidate/include')
            inherited=dict(os.environ);calls=[]
            def output(argv,**kwargs):
                self.assertIs(kwargs['env'],env);calls.append(argv)
                return 'object: '+str(code)+'\n' if '-M' in argv else 'toolchain'
            def compile(argv,**kwargs):
                self.assertIs(kwargs['env'],env);calls.append(argv)
                Path(argv[-1]).write_bytes(b'compiled')
                return type('Exit',(),{'returncode':0})()
            with patch.object(builder,'SOURCES',['candidate.cu']),patch.object(builder.subprocess,'check_output',side_effect=output), \
                 patch.object(builder.subprocess,'run',side_effect=compile):
                self.assertIn(code,builder.freeze_inputs(source,'nvcc',env=env))
                result=builder.build(source,root/'cache','nvcc',env=env)
            self.assertEqual(sum('-M' in call for call in calls),2)
            self.assertTrue(any('-c' in call for call in calls))
            self.assertTrue(any('--shared' in call for call in calls))
            manifest=json.loads((Path(result['path']).parent/'manifest.json').read_text())
            self.assertEqual(manifest['inputs']['toolchain']['environment']['CPATH'],'/candidate/include')
            self.assertEqual(dict(os.environ),inherited)

    def test_existing_disk_native_cache_is_not_replaced(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                root = Path(directory).resolve();(root/'kernel').mkdir()
                native = root/'kernel/build'
                if symlink:
                    other = root/'other-disk';other.mkdir();native.symlink_to(other, target_is_directory=True)
                else:
                    native.mkdir()
                resolved = native.resolve();(native/'sentinel').write_text('keep')
                with patch.object(host, 'filesystem_type', return_value='nfs4'):
                    host.prepare_build(root, root/'configured')
                self.assertEqual(native.resolve(), resolved)
                self.assertEqual(native.is_symlink(), symlink)
                self.assertEqual((native/'sentinel').read_text(), 'keep')

    def test_existing_disk_objects_link_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve();(root/'kernel').mkdir()
            build = root/'configured';(build/'candidates').mkdir(parents=True)
            disk = root/'other-disk';disk.mkdir();(disk/'sentinel').write_text('keep')
            objects = build/'candidates/objects';objects.symlink_to(disk, target_is_directory=True)
            with patch.object(host, 'filesystem_type', return_value='ext4') as filesystem:
                host.prepare_build(root, build)
            self.assertIn(unittest.mock.call(disk), filesystem.call_args_list)
            self.assertTrue(objects.is_symlink())
            self.assertEqual(objects.resolve(), disk)
            self.assertEqual((objects/'sentinel').read_text(), 'keep')

    def test_ram_cache_or_hidden_native_link_fails_before_creating_build_root(self):
        for location in ('root', 'target', 'tmp', 'candidates', 'candidates/objects', 'native'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve();(root/'kernel').mkdir()
                build = root/'configured'
                ram = root/'ram';ram.mkdir()
                if location == 'native':
                    (root/'kernel/build').symlink_to(ram, target_is_directory=True)
                    bad = ram
                elif location == 'root':
                    bad = build
                else:
                    (build/location).parent.mkdir(parents=True)
                    (build/location).symlink_to(ram, target_is_directory=True)
                    bad = ram
                def filesystem(path):
                    return 'tmpfs' if path == bad else 'ext4'
                before = set(root.rglob('*'))
                with patch.object(host, 'filesystem_type', side_effect=filesystem):
                    with self.assertRaisesRegex(ValueError, 'disk-backed.*local SSD'):
                        host.prepare_build(root, build)
                self.assertEqual(set(root.rglob('*')), before)
                if location == 'native':
                    self.assertEqual((root/'kernel/build').resolve(), ram)

    def test_broken_native_link_requires_explicit_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve();(root/'kernel').mkdir()
            (root/'kernel/build').symlink_to(root/'missing', target_is_directory=True)
            with patch.object(host, 'filesystem_type', return_value='ext4'):
                with self.assertRaisesRegex(ValueError, 'broken link'):
                    host.prepare_build(root, root/'configured')
            self.assertFalse((root/'configured').exists())


if __name__ == '__main__':
    unittest.main()
