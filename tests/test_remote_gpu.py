"""Source isolation and cross-shell argument transport for the GPU launcher."""
import base64
import os
import re
import importlib.machinery
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/gpu'
loader = importlib.machinery.SourceFileLoader('remote_gpu', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
gpu = importlib.util.module_from_spec(spec)
loader.exec_module(gpu)


class RemoteGpuTests(unittest.TestCase):
    def test_snapshot_preserves_dirty_source_excludes_ignored_and_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check_snapshot(Path(directory))

    def check_snapshot(self, tmp_path):
        def git(*args):
            subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
        git('init')
        (tmp_path / '.gitignore').write_text('private/\n*.pt\n')
        (tmp_path / 'source.py').write_text('old')
        (tmp_path / 'deleted.py').write_text('old')
        git('add', '.')
        (tmp_path / 'source.py').write_text('edited')
        (tmp_path / 'deleted.py').unlink()
        (tmp_path / 'new.py').write_text('new')
        (tmp_path / 'private').mkdir()
        (tmp_path / 'private/video.mp4').write_bytes(b'private')
        (tmp_path / 'model.pt').write_bytes(b'weights')
        with patch.object(gpu, 'ROOT', tmp_path):
            with tempfile.TemporaryFile() as output:
                gpu.snapshot(output)
                output.seek(0)
                with tarfile.open(fileobj=output) as archive:
                    assert set(archive.getnames()) == {'.gitignore', 'source.py', 'new.py'}
                    assert archive.extractfile('source.py').read() == b'edited'

    def test_snapshot_uses_linked_worktree_and_its_initialized_submodule(self):
        def git(root, *args):
            subprocess.run(
                ['git', '-C', str(root), *args],
                check=True,
                capture_output=True,
            )

        def commit(root):
            git(root, 'add', '.')
            git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                'commit', '-m', 'Initialize repository')

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            tracker = tmp_path / 'tracker'
            checkout = tmp_path / 'checkout'
            worktree = checkout / '.worktrees' / 'feature'
            for root in [tracker, checkout]:
                root.mkdir()
                git(root, 'init')
            (tracker / 'tracker.py').write_text('tracker')
            (tracker / '.gitignore').write_text('ckpts/\n')
            commit(tracker)
            (checkout / '.gitignore').write_text('.worktrees/\n')
            (checkout / 'pipeline.py').write_text('base')
            git(checkout, '-c', 'protocol.file.allow=always', 'submodule', 'add',
                str(tracker), 'src/TrackNetV3')
            commit(checkout)
            git(checkout, 'worktree', 'add', '-b', 'feature', str(worktree))
            git(worktree, '-c', 'protocol.file.allow=always',
                'submodule', 'update', '--init')
            (checkout / 'pipeline.py').write_text('main edit')
            (checkout / 'main_only.py').write_text('main only')
            (worktree / 'pipeline.py').write_text('feature edit')
            tracknet = worktree / 'src/TrackNetV3'
            assert (tracknet / '.git').is_file()
            (tracknet / 'tracker.py').write_text('feature tracker edit')
            (tracknet / 'ckpts').mkdir()
            (tracknet / 'ckpts/weights.pt').write_bytes(b'weights')

            with patch.object(gpu, 'ROOT', worktree), tempfile.TemporaryFile() as output:
                gpu.snapshot(output)
                output.seek(0)
                with tarfile.open(fileobj=output) as archive:
                    assert set(archive.getnames()) == {
                        '.gitignore', '.gitmodules', 'pipeline.py',
                        'src/TrackNetV3/.gitignore', 'src/TrackNetV3/tracker.py',
                    }
                    assert archive.extractfile('pipeline.py').read() == b'feature edit'
                    assert archive.extractfile('src/TrackNetV3/tracker.py').read() == b'feature tracker edit'

    def test_remote_python_encodes_arguments_without_shell_interpretation(self):
        args = ['~/code/with spaces', 'quotes " and \' ; $HOME %PATH%\nnewline']
        captured = {}
        def record(host, command, **kwargs):
            captured.update(host=host, command=command)
        with patch.dict(os.environ, {'BV_GPU_TRANSPORT': 'native'}), patch.object(gpu, 'ssh', record):
            gpu.remote_python('avalanche', 'assert sys.argv[1:] == ' + repr(args), *args)
        words = shlex.split(captured['command'])
        assert words[:2] == ['python3', '-c']
        assert args[1] not in captured['command']
        # Execute only the generated Python payload, with argv restored afterwards.
        with patch.object(sys, 'argv', ['test']):
            exec(words[2], {'base64': base64})


    def test_wsl_transport_preserves_arguments_without_nested_windows_quotes(self):
        args = ['spaces ; $HOME %PATH% "quotes"', 'line\nnext']
        with patch.dict(os.environ, {'BV_GPU_TRANSPORT': 'wsl'}), patch.object(gpu.subprocess, 'run') as run:
            gpu.remote_python('avalanche', 'assert sys.argv[1:] == ' + repr(args), *args)
        command = run.call_args.args[0][-1]
        self.assertTrue(command.startswith('wsl.exe --cd ~ --exec python3 -c '))
        self.assertNotIn('"', command)
        payload = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", command).group(1)
        with patch.object(sys, 'argv', ['test']):
            exec(base64.b64decode(payload), {})

    def test_shell_transport_encodes_metacharacters(self):
        original = "printf '%s' '$HOME & 100%'"
        with patch.dict(os.environ, {'BV_GPU_TRANSPORT': 'wsl'}), patch.object(gpu.subprocess, 'run') as run:
            gpu.ssh('avalanche', original)
        command = run.call_args.args[0][-1]
        self.assertNotIn(original, command)
        self.assertNotIn('"', command)
        payload = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", command).group(1)
        with patch('os.execl') as execute:
            exec(base64.b64decode(payload), {})
        execute.assert_called_once_with('/bin/bash', 'bash', '-lc', original)


    def test_windows_job_uses_existing_account_and_foreground_wsl(self):
        with patch.object(gpu, 'ssh') as send:
            gpu.launch_windows_job('avalanche', '/srv/job with spaces', '20261118T120000-1234abcd')
        command = send.call_args.args[1]
        self.assertTrue(send.call_args.kwargs['windows_command'])
        encoded = command.split(' -EncodedCommand ', 1)[1]
        script = base64.b64decode(encoded).decode('utf-16le')
        self.assertIn("GetCurrent().Name", script)
        self.assertIn('-LogonType S4U -RunLevel Limited', script)
        self.assertIn('/bin/bash', script)
        self.assertIn('/srv/job with spaces/run.sh', script)
        self.assertIn("throw 'Task already exists'", script)
        self.assertIn('Start-ScheduledTask', script)


if __name__ == "__main__":
    unittest.main()
