"""Never expose a partial recording or silently replace existing user data."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f'Output locked: {path}; if interrupted, verify no writer is running '
                              'before manually removing its stale lock') from exc
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(f'pid={os.getpid()}\n')
        yield
    finally:
        path.unlink()


@contextmanager
def staged_recording(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f'.{destination.name}.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(f'pid={os.getpid()}\n')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Refusing existing recording destination: {destination}')
        with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.staging-', dir=destination.parent) as temporary:
            stage = Path(temporary) / 'recording'
            stage.mkdir()
            yield stage
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f'Destination appeared during conversion: {destination}')
            stage.rename(destination)
    finally:
        lock.unlink()


def atomic_write_text(destination, text):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_npz_exclusive(destination, **arrays):
    import numpy as np
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing existing NPZ destination: {destination}')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)  # Atomic publication without replacement.
    finally:
        Path(temporary).unlink(missing_ok=True)
