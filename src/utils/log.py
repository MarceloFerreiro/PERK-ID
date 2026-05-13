
import threading
import sys

_VERBOSE = False
_WORKERS = 0
_WORKER_ID = threading.local()
_PRINT_LOCK = threading.Lock()  # Lock to serialize stdout access
_TQDM_WRITE = None  # Reference to tqdm.write function


def set_log_context(verbose: bool, workers: int) -> None:
    global _VERBOSE, _WORKERS
    _VERBOSE = verbose
    _WORKERS = workers


def set_worker_id(worker_id: int) -> None:
    """Set the current worker ID (thread-local)."""
    _WORKER_ID.id = worker_id


def set_tqdm_write(write_func) -> None:
    """Set the tqdm.write function for thread-safe output."""
    global _TQDM_WRITE
    _TQDM_WRITE = write_func


def _get_worker_id_str() -> str:
    """Get the current worker ID as a string, or 'main' if not set."""
    return str(getattr(_WORKER_ID, 'id', 'main'))


def _safe_print(msg: str) -> None:
    """Thread-safe print using tqdm.write if available, otherwise with lock."""
    if _TQDM_WRITE is not None:
        _TQDM_WRITE(msg)
    else:
        with _PRINT_LOCK:
            print(msg, file=sys.stdout, flush=True)


def _log_timed(elapsed: float, feature: str, extra: str) -> None:
    if not _VERBOSE:
        return
    worker_id = _get_worker_id_str()
    _safe_print(f"{elapsed:.2f}s [w{worker_id}] {feature}: {extra}")
