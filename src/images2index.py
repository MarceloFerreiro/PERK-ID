import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from time import time
import threading

from tqdm import tqdm

from src.features import features

# --- Placeholder feature functions ---

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
    
    # Assign a unique worker ID to this thread
    with _WORKER_COUNTER_LOCK:
        worker_id = _WORKER_COUNTER
        _WORKER_COUNTER += 1
    set_worker_id(worker_id)

def _save_partial(out_path: Path, paths: list[Path], results: dict[Path, np.ndarray]) -> None:
    """Save partial results. Check for shape inconsistencies."""
    completed_paths = [p for p in paths if p in results]
    if not completed_paths:
        return
    
    # Debug: check shapes
    shapes = [results[p].shape for p in completed_paths]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        _safe_print(f"[!] WARNING: Inconsistent shapes found: {unique_shapes}")
        shape_counts = {}
        bad_paths = []
        for i, p in enumerate(completed_paths):
            s = results[p].shape
            shape_counts[s] = shape_counts.get(s, 0) + 1
            if s != shapes[0]:
                bad_paths.append((i, p, s))
        
        for i, p, s in bad_paths[:10]:  # Show first 10 problematic paths
            _safe_print(f"    Path {i}: {p.name} has shape {s} (expected {shapes[0]})")
        if len(bad_paths) > 10:
            _safe_print(f"    ... and {len(bad_paths) - 10} more")
        
        _safe_print(f"[*] Shape distribution: {shape_counts}")
        raise ValueError(f"Feature vectors have inconsistent shapes: {unique_shapes}. All should be {shapes[0]}")
    
    partial_matrix = np.stack([results[p] for p in completed_paths])
    np.savez(out_path, matrix=partial_matrix, paths=[str(p) for p in completed_paths])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("data/images"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/features.npz"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--stalled-secs", type=int, default=30,
                        help="Log a warning if no image finishes in N seconds")
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
        futures = {pool.submit(features, p): p for p in paths}
        with tqdm(
            total=len(paths),
            unit="imagenes",
            postfix="Extraccion",
            #disable=not _VERBOSE,
        ) as pbar:
            # Use tqdm.write for thread-safe logging
            set_tqdm_write(pbar.write)
            
            pending = set(futures)
            processed = 0
            try:
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=max(1, args.stalled_secs),
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
                            if _VERBOSE:
                                _safe_print(f"[!] Failed {path}: {exc}")
                            pbar.update(1)
                            processed += 1
                            continue
                        
                        if result is None:
                            skipped.append(path)
                            pbar.set_description(f'fracasos: {len(skipped)}')
                            if _VERBOSE:
                                _safe_print(f"[!] Skipped unreadable image: {path}")
                            pbar.update(1)
                            processed += 1
                            continue
                        
                        results[path] = result
                        pbar.set_description(f'exitos: {len(results)}')
                        pbar.update(1)
                        processed += 1
                        
                        # Save partial results only from main thread (worker_id='main')
                        if args.save_every > 0 and args.out and len(results) % args.save_every == 0:
                            try:
                                _safe_print(f"[*] Saving partial: {len(results)} results")
                                _save_partial(args.out, paths, results)
                                _safe_print(f"[*] Saved {len(results)} results to {args.out}")
                            except Exception as save_exc:
                                _safe_print(f"[!] Save failed at {len(results)} results: {save_exc}")
                                import traceback
                                _safe_print(traceback.format_exc())
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
    
    # Check for shape consistency
    shapes = [r.shape for r in ordered]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        _safe_print(f"[!] ERROR: Inconsistent shapes in final results: {unique_shapes}")
        for i, (p, r) in enumerate(zip(completed_paths, ordered)):
            if r.shape != shapes[0]:
                _safe_print(f"    Image {i}: {p.name} has shape {r.shape} (expected {shapes[0]})")
        raise ValueError(f"Feature vectors have inconsistent shapes: {unique_shapes}")
    
    matrix = np.stack(ordered)
    paths = completed_paths
    np.savez(args.out, matrix=matrix, paths=[str(p) for p in paths])
    _log_timed(time() - start, "MATRIX", f"{matrix.shape} ({len(paths)} images)")


if __name__ == "__main__":
    main()
