"""Opt-in end-to-end test of the Laptop entry point and real 5080 session.

IGNIS_LAB_E2E_CONFIG=/absolute/lab-settings.json python3 -m unittest discover -s tests -p test_lab_gpu.py
The configuration may explicitly cap diagnostic output and capture scope; the
result is a functional regression, never a production benchmark verdict.
"""
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('IGNIS_LAB_E2E_CONFIG'), 'opt-in exclusive RTX 5080 integration')
class LabGpuTests(unittest.TestCase):
    def command(self,*args,ok=True):
        p=subprocess.run([sys.executable,str(ROOT/'scripts/lab.py'),*args],cwd=ROOT,
                         capture_output=True,text=True,timeout=5400)
        self.logs.append(dict(args=args,code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        (self.output/'commands.json').write_text(json.dumps(self.logs,indent=2))
        self.assertEqual(p.returncode==0,ok,p.stderr+'\n'+p.stdout[-2000:])
        return json.loads(p.stdout) if p.stdout.strip() else {}

    def test_capture_reload_compile_failure_and_persistent_reuse(self):
        self.output=ROOT/'.scratch/lab-e2e';self.output.mkdir(exist_ok=True);self.logs=[]
        config=str(Path(os.environ['IGNIS_LAB_E2E_CONFIG']).resolve())
        source=ROOT/'kernel/src/trellis_linear.cu';original=source.read_text()
        try:
            self.command('stop')
            started=self.command('setup','--config',config,'--new-baseline')
            pid=started['engine_pid'];corpus=started['corpus']
            good=self.command('replay')
            self.assertTrue(good['passed']);self.assertEqual(good['engine_pid'],pid)
            # A real edit, a new immutable library and the same resident model.
            source.write_text(original+'\n// resident-lab candidate identity regression\n')
            changed=self.command('replay')
            self.assertTrue(changed['passed']);self.assertEqual(changed['engine_pid'],pid)
            self.assertNotEqual(good['candidate']['identity'],changed['candidate']['identity'])
            source.write_text(original+'\nTHIS_IS_AN_INTENTIONAL_COMPILE_FAILURE\n')
            self.command('replay',ok=False)
            self.assertEqual(self.command('setup')['engine_pid'],pid)
            # Deliberately wrong, graph-capturable computation: reload must take
            # effect and the real captured-data gate must reject it.
            at=original.index('extern "C" int32_t ignis_trellis_linear_swiglu(')
            brace=original.index('{',at)+1
            wrong='\n  return cudaMemset2DAsync(h,2ull*ldh,0,2ull*gate->out,tokens,static_cast<cudaStream_t>(stream))==cudaSuccess?0:-1;\n'
            source.write_text(original[:brace]+wrong+original[brace:])
            rejected=self.command('replay',ok=False)
            self.assertFalse(rejected['passed']);self.assertEqual(rejected['engine_pid'],pid)
            self.assertFalse(rejected['replay']['passed'])
            source.write_text(original)
            restored=self.command('forward')
            self.assertTrue(restored['passed']);self.assertEqual(restored['engine_pid'],pid)
            self.command('stop');self.command('stop')
            restarted=self.command('setup')
            self.assertNotEqual(restarted['engine_pid'],pid)
            self.assertEqual(restarted['corpus'],corpus)
            self.assertTrue(self.command('replay')['passed'])
        finally:
            source.write_text(original)
            self.command('stop')

    def test_disconnect_busy_stop_and_owner_death(self):
        self.output=ROOT/'.scratch/lab-lifecycle';self.output.mkdir(exist_ok=True);self.logs=[]
        config=str(Path(os.environ['IGNIS_LAB_E2E_CONFIG']).resolve())
        def remote(code,*args):
            return json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/remote.py'),
                'exec','--no-env','--','/usr/bin/python3','-c',code,*args],cwd=ROOT,text=True))
        def state():
            return remote("import sys,json;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import status;print(json.dumps(status(Path.cwd())))")
        def wait(predicate,limit=180):
            deadline=time.monotonic()+limit
            while time.monotonic()<deadline:
                current=state()
                if predicate(current):return current
                time.sleep(.5)
            self.fail('lifecycle deadline: '+str(current))
        def start(action,label):
            out=(self.output/(label+'.out')).open('w');err=(self.output/(label+'.err')).open('w')
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/lab.py'),action],cwd=ROOT,
                stdout=out,stderr=err,start_new_session=True)
            out.close();err.close();return p
        process=None
        try:
            ready=self.command('setup','--config',config)
            pid=ready['engine_pid']
            process=start('replay','disconnected')
            active=wait(lambda s:s['status']=='busy');operation=active['active']
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20);process=None
            done=wait(lambda s:s['status']=='ready')
            self.assertEqual(done['engine_pid'],pid)
            result=remote("import json,sys;from pathlib import Path;print(Path(sys.argv[1]).read_text())",done['last_result'])
            self.assertTrue(result['finished']);self.assertTrue(result['passed'])
            self.assertEqual(result['operation'],operation)
            process=start('forward','stopped-busy')
            wait(lambda s:s['status']=='busy');time.sleep(1)
            self.command('stop')
            self.assertNotEqual(process.wait(timeout=30),0);process=None
            self.assertEqual(state()['status'],'stopped')
            ready=self.command('setup')
            current=state()
            remote("import json,sys,os,signal;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import identity; s=json.loads(Path('.scratch/lab/state.json').read_text()); assert identity(s['process']['pid'])==s['process'];os.kill(s['process']['pid'],signal.SIGKILL);print('{}')")
            wait(lambda s:s['status']=='stopped',60)
            gone=remote("import json,sys;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import read,alive;print(json.dumps(not alive(read(Path(sys.argv[1])/'engine.identity.json'))))",current['directory'])
            self.assertTrue(gone,'owner death leaked the GPU child')
            self.command('stop')
            (self.output/'completed.json').write_text(json.dumps(dict(passed=True,disconnected_operation=operation)))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20)
            self.command('stop')
