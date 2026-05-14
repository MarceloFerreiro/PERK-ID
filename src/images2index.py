import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from time import time
import threading

from tqdm import tqdm

from src.features import features, read_image

_WORKERS = 0
_VERBOSE = False
_WORKER_COUNTER = 0
_WORKER_COUNTER_LOCK = threading.Lock()


from src.utils.log import _log_timed, set_log_context, set_worker_id, set_tqdm_write, _safe_print

def _init_worker(verbose: bool, workers: int) -> None:
    global _VERBOSE, _WORKERS, _WORKER_COUNTER
    _VERBOSE = verbose
    _WORKERS = workers
    set_log_context(verbose, workers)
    
    with _WORKER_COUNTER_LOCK:
        worker_id = _WORKER_COUNTER
        _WORKER_COUNTER += 1
    set_worker_id(worker_id)

def _save_partial(out_path: Path, paths: list[Path], results: dict[Path, np.ndarray]) -> None:
    """Save partial results. Check for shape inconsistencies."""
    completed_paths = [p for p in paths if p in results]
    if not completed_paths:
        return
    
    partial_matrix = np.stack([results[p] for p in completed_paths])
    np.savez(out_path, matrix=partial_matrix, paths=[str(p) for p in completed_paths])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("data/images"))
    parser.add_argument("--workers", type=int, default=8, help='Número de hilos a usar')
    parser.add_argument("--out", type=Path, default=Path("data/features.npz"), help='Ruta al índice .npz')
    parser.add_argument("--save-every", type=int, default=10, help='Cada cuantas imágenes se actualiza el indice en disco')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _VERBOSE = args.verbose
    start = time()
    global _WORKERS
    _WORKERS = args.workers
    set_log_context(_VERBOSE, _WORKERS)
    paths = sorted(args.dir.glob("*.jp*g"))  # matches .jpg and .jpeg

    results = {}
    skipped: list[Path] = []
    with ThreadPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(_VERBOSE, args.workers),
    ) as pool:
        futures = {pool.submit(lambda p: features(read_image(p)), p): p for p in paths}
        with tqdm(
            total=len(paths),
            unit="imagenes",
            postfix="Extraccion",
        ) as pbar:
            set_tqdm_write(pbar.write)
            
            pending = set(futures)
            processed = 0
            try:
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=max(1, 10),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        if _VERBOSE:
                            _safe_print(f"[!] No progress in {args.stalled_secs}s; pending={len(pending)}")
                        continue
                    
                    for future in done:
                        path = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            skipped.append(path)
                            if _VERBOSE: _safe_print(f"[!] Failed {path}: {exc}")
                            pbar.update(1)
                            processed += 1
                            continue
                        
                        results[path] = result
                        pbar.set_description(f'exitos: {len(results)}')
                        pbar.update(1)
                        processed += 1
                        
                        if args.save_every > 0 and args.out and len(results) % args.save_every == 0:
                            _safe_print(f"[*] Saving partial: {len(results)} results")
                            _save_partial(args.out, paths, results)
                            _safe_print(f"[*] Saved {len(results)} results to {args.out}")
            except Exception as loop_exc:
                _safe_print(f"[!] Loop exception at processed={processed}: {loop_exc}")
                import traceback
                _safe_print(traceback.format_exc())
                raise
            finally:
                _safe_print(f"[*] Loop completed: processed={processed}, results={len(results)}")

    completed_paths = [p for p in paths if p in results]
    if not completed_paths:
        raise RuntimeError("No images could be processed successfully.")
    ordered = [results[p] for p in completed_paths]
    set_tqdm_write(None)  # Reset tqdm.write before final save
    if skipped and _VERBOSE:
        _safe_print(f"[!] Skipped {len(skipped)} images due to read errors.")
    
    shapes = [r.shape for r in ordered]
    unique_shapes = set(shapes)
    
    matrix = np.stack(ordered)
    paths = completed_paths
    np.savez(args.out, matrix=matrix, paths=[str(p) for p in paths])
    _log_timed(time() - start, "MATRIX", f"{matrix.shape} ({len(paths)} images)")


if __name__ == "__main__":
    main()
