"""Crash-recoverable publication of one stage's output bundle and receipt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _atomic_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _transaction_directory(receipt: Path) -> Path:
    return receipt.with_name(f".{receipt.name}.publication")


@contextmanager
def _lock(receipt: Path):
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lock = receipt.with_name(f".{receipt.name}.publication.lock")
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"stage publication is already running: {receipt}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup(transaction: Path) -> None:
    shutil.rmtree(transaction)
    _sync_directory(transaction.parent)


def _recover(receipt: Path) -> bool:
    transaction = _transaction_directory(receipt)
    if not transaction.exists():
        return False
    journal_path = transaction / "journal.json"
    if not journal_path.exists():
        # Preparation never changes canonical paths before publishing this journal.
        _cleanup(transaction)
        return True
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("schema") != "stage_publication" or journal.get("receipt") != str(
        receipt
    ):
        raise ValueError(f"invalid stage publication journal: {journal_path}")
    if journal.get("state") in {"committed", "rolled_back"}:
        _cleanup(transaction)
        return True
    committed = receipt.is_file() and _digest(receipt) == journal["new_receipt_sha256"]
    if committed:
        committed = all(
            Path(output["path"]).is_file()
            and _digest(Path(output["path"])) == output["new_sha256"]
            for output in journal["outputs"]
        )
    if committed:
        journal["state"] = "committed"
        _atomic_bytes(journal_path, _json_bytes(journal))
        _cleanup(transaction)
        return True
    # Backups are copied, never consumed: interruption during rollback is safe.
    for output in [*journal["outputs"], journal["previous_receipt"]]:
        destination = Path(output["path"])
        if output["existed"]:
            backup = transaction / output["backup"]
            if not backup.is_file() or _digest(backup) != output["previous_sha256"]:
                raise ValueError(
                    f"stage publication backup is missing or corrupt: {backup}"
                )
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.restore.", dir=destination.parent
            )
            os.close(descriptor)
            temporary = Path(name)
            try:
                _copy(backup, temporary)
                temporary.replace(destination)
                _sync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            destination.unlink(missing_ok=True)
            _sync_directory(destination.parent)
    journal["state"] = "rolled_back"
    _atomic_bytes(journal_path, _json_bytes(journal))
    _cleanup(transaction)
    return True


def recover(receipt_path: str | Path) -> bool:
    """Restore the last completed bundle, or finish cleanup after receipt commit.

    Call before inspecting a stage receipt or reading its canonical outputs.
    Returns whether an interrupted publication was found and recovered.
    """
    receipt = Path(receipt_path).expanduser().resolve()
    with _lock(receipt):
        return _recover(receipt)


def publish(
    staged: Sequence[Path],
    outputs: Sequence[Path],
    receipt_path: str | Path,
    receipt_value: Mapping[str, Any],
) -> Path:
    """Publish validated files, retaining recoverable backups until receipt commit.

    A process killed between file replacements leaves a durable journal. The next
    ``recover`` rolls back the whole bundle, including the old receipt. Outputs
    with no prior version are removed on rollback. No annotations are touched
    unless explicitly passed as a stage output.
    """
    receipt = Path(receipt_path).expanduser().resolve()
    destinations = [Path(path).expanduser().resolve() for path in outputs]
    sources = [Path(path).expanduser().resolve() for path in staged]
    if not sources or len(sources) != len(destinations):
        raise ValueError("publication needs matching nonempty staged/output lists")
    if len(set([*destinations, receipt])) != len(destinations) + 1:
        raise ValueError("publication output and receipt paths must be distinct")
    new_receipt = _json_bytes(receipt_value)
    with _lock(receipt):
        _recover(receipt)
        transaction = _transaction_directory(receipt)
        transaction.mkdir()
        _sync_directory(transaction.parent)
        try:
            previous = []
            for index, destination in enumerate([*destinations, receipt]):
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = f"previous-{index}"
                entry = {
                    "path": str(destination),
                    "existed": destination.exists(),
                    "backup": backup,
                }
                if entry["existed"]:
                    _copy(destination, transaction / backup)
                    entry["previous_sha256"] = _digest(transaction / backup)
                previous.append(entry)
            for index, source in enumerate(sources):
                prepared = transaction / f"next-{index}"
                _copy(source, prepared)
                previous[index]["prepared"] = prepared.name
                previous[index]["new_sha256"] = _digest(prepared)
            journal = {
                "schema": "stage_publication",
                "schema_version": 1,
                "state": "prepared",
                "receipt": str(receipt),
                "new_receipt_sha256": hashlib.sha256(new_receipt).hexdigest(),
                "outputs": previous[:-1],
                "previous_receipt": previous[-1],
            }
            _atomic_bytes(transaction / "journal.json", _json_bytes(journal))
            for output in journal["outputs"]:
                destination = Path(output["path"])
                (transaction / output["prepared"]).replace(destination)
                _sync_directory(destination.parent)
            _atomic_bytes(receipt, new_receipt)
        except BaseException:
            _recover(receipt)
            raise
        _recover(receipt)
    return receipt
