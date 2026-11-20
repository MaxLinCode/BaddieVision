"""A failed or killed publication must preserve the previous usable bundle."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.workflow.publication import publish, recover


def _bundle(tmp_path):
    staged = [tmp_path / "staged-a", tmp_path / "staged-b"]
    outputs = [tmp_path / "a.jsonl", tmp_path / "b.csv"]
    for index, path in enumerate(staged):
        path.write_text(f"new-{index}")
    for index, path in enumerate(outputs):
        path.write_text(f"old-{index}")
    receipt = tmp_path / "manifest.json"
    receipt.write_bytes(b'{"old_receipt": true}\n')
    return staged, outputs, receipt


def test_successful_publication_commits_complete_bundle_and_receipt(tmp_path):
    staged, outputs, receipt = _bundle(tmp_path)

    assert publish(staged, outputs, receipt, {"revision": 2}) == receipt

    assert [path.read_text() for path in outputs] == ["new-0", "new-1"]
    assert json.loads(receipt.read_text()) == {"revision": 2}
    assert not recover(receipt)
    assert all(path.exists() for path in staged)


def test_failed_second_replacement_restores_previous_outputs_and_exact_receipt(tmp_path, monkeypatch):
    staged, outputs, receipt = _bundle(tmp_path)
    previous_receipt = receipt.read_bytes()
    replace = Path.replace
    failed = False

    def fail_second(path, destination):
        nonlocal failed
        if Path(destination) == outputs[1] and not failed:
            failed = True
            raise OSError("second replacement failed")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError, match="second replacement failed"):
        publish(staged, outputs, receipt, {"revision": 2})

    assert [path.read_text() for path in outputs] == ["old-0", "old-1"]
    assert receipt.read_bytes() == previous_receipt
    assert not recover(receipt)


def test_failure_before_receipt_commit_removes_outputs_without_previous_versions(tmp_path, monkeypatch):
    staged, outputs, receipt = _bundle(tmp_path)
    outputs[1].unlink()
    receipt.unlink()
    replace = Path.replace
    failed = False

    def fail_receipt(path, destination):
        nonlocal failed
        if Path(destination) == receipt and not failed:
            failed = True
            raise OSError("receipt commit failed")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_receipt)
    with pytest.raises(OSError, match="receipt commit failed"):
        publish(staged, outputs, receipt, {"revision": 2})

    assert outputs[0].read_text() == "old-0"
    assert not outputs[1].exists()
    assert not receipt.exists()


@pytest.mark.parametrize("point", ["first_output", "second_output", "receipt"])
def test_next_process_recovers_abrupt_exit_mid_publication(tmp_path, point):
    staged, outputs, receipt = _bundle(tmp_path)
    previous_receipt = receipt.read_bytes()
    script = r'''
import os, sys
from pathlib import Path
from src.workflow.publication import publish
directory, point = Path(sys.argv[1]), sys.argv[2]
target = {"first_output": "a.jsonl", "second_output": "b.csv", "receipt": "manifest.json"}[point]
replace = Path.replace
def interrupted_replace(path, destination):
    result = replace(path, destination)
    if Path(destination).name == target:
        os._exit(17)
    return result
Path.replace = interrupted_replace
publish([directory / "staged-a", directory / "staged-b"],
        [directory / "a.jsonl", directory / "b.csv"],
        directory / "manifest.json", {"revision": 2})
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path), point], check=False)
    assert process.returncode == 17

    assert recover(receipt)

    if point == "receipt":
        assert [path.read_text() for path in outputs] == ["new-0", "new-1"]
        assert json.loads(receipt.read_text()) == {"revision": 2}
    else:
        assert [path.read_text() for path in outputs] == ["old-0", "old-1"]
        assert receipt.read_bytes() == previous_receipt
    assert not recover(receipt)


def test_interrupted_rollback_can_itself_be_recovered(tmp_path):
    staged, outputs, receipt = _bundle(tmp_path)
    script = r'''
import os, sys
from pathlib import Path
from src.workflow.publication import publish
directory = Path(sys.argv[1])
replace = Path.replace
replacements = 0
def interrupted_replace(path, destination):
    global replacements
    if Path(destination).name == "b.csv":
        raise OSError("start rollback")
    result = replace(path, destination)
    if Path(destination).name == "a.jsonl":
        replacements += 1
        if replacements == 2:
            os._exit(18)
    return result
Path.replace = interrupted_replace
publish([directory / "staged-a", directory / "staged-b"],
        [directory / "a.jsonl", directory / "b.csv"],
        directory / "manifest.json", {"revision": 2})
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert process.returncode == 18

    assert recover(receipt)

    assert [path.read_text() for path in outputs] == ["old-0", "old-1"]
    assert json.loads(receipt.read_text()) == {"old_receipt": True}


def test_interrupted_preparation_never_changes_canonical_outputs(tmp_path):
    _, outputs, receipt = _bundle(tmp_path)
    abandoned = receipt.with_name(f".{receipt.name}.publication")
    abandoned.mkdir()
    (abandoned / "incomplete-backup").write_bytes(b"partial")

    assert recover(receipt)

    assert [path.read_text() for path in outputs] == ["old-0", "old-1"]
    assert json.loads(receipt.read_text()) == {"old_receipt": True}
