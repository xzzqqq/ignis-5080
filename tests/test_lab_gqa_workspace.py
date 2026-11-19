"""CPU checks for candidate-owned GQA retained workspace contracts."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GqaWorkspaceTests(unittest.TestCase):
    def test_split_and_interleaved_hq_retention(self):
        compiler = os.environ.get('CXX') or shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            executable = str(Path(tmp) / 'gqa-workspace')
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I' + str(ROOT / 'kernel/include'), '-I' + str(ROOT / 'kernel/src'),
                            str(ROOT / 'kernel/tests/test_gqa_segment_workspace.cpp'),
                            '-o', executable], check=True)
            subprocess.run([executable], check=True)


if __name__ == '__main__':
    unittest.main()
