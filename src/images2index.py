import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from time import time
import threading

from tqdm import tqdm

from src.features import features, read_image, extract_sift_descriptors
from src.features.BoW import BoWExtractor

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

def _process_image(image_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Process a single image to extract features and SIFT descriptors.
    
    Returns:
        Tuple of (path, features_vector, sift_descriptors)
    """
    try:
        image = read_image(image_path)
        feat_vector = features(image, include_sift=False)
        sift_desc = extract_sift_descriptors(image)
        return (image_path, feat_vector, sift_desc)
    except Exception as e:
        raise RuntimeError(f"Failed to process {image_path}: {e}") from e

def _save_partial(out_path: Path, paths: list[Path], feature_vectors: dict[Path, np.ndarray]) -> None:
    """Save partial results."""
    completed_paths = [p for p in paths if p in feature_vectors]
    if not completed_paths:
        return
    
    partial_matrix = np.stack([feature_vectors[p] for p in completed_paths])
    np.savez(out_path, matrix=partial_matrix, paths=[str(p) for p in completed_paths])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("data/images"))
    parser.add_argument("--workers", type=int, default=8, help='Número de hilos a usar')
    parser.add_argument("--out", type=Path, default=Path("data/features.npz"), help='Ruta al índice .npz')
    parser.add_argument("--bow", type=Path, default=Path("data/bow.pkl"), help='Ruta para guardar el modelo BoW')
    parser.add_argument("--save-every", type=int, default=10, help='Cada cuantas imágenes se actualiza el indice en disco')
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1, help="Número máximo de imágenes a procesar")
    args = parser.parse_args()

    _VERBOSE = args.verbose
    start = time()
    global _WORKERS
    _WORKERS = args.workers
    set_log_context(_VERBOSE, _WORKERS)
    paths = sorted(list(args.dir.glob("*.jpg")) +
                   list(args.dir.glob("*.jpeg")) +
                   list(args.dir.glob("*.png")))[:args.max_images]
    feature_vectors = {}  # path -> feature vector
    sift_descriptors_by_path = {}  # path -> sift descriptors
    all_sift_descriptors = []  # collect all descriptors for KMeans
    skipped: list[Path] = []
    
    with ThreadPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(_VERBOSE, args.workers),
    ) as pool:
        futures = {pool.submit(_process_image, p): p for p in paths}
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
                            result_path, feat_vec, sift_desc = future.result()
                        except Exception as exc:
                            skipped.append(path)
                            if _VERBOSE: _safe_print(f"[!] Failed {path}: {exc}")
                            pbar.update(1)
                            processed += 1
                            continue
                        
                        feature_vectors[result_path] = feat_vec
                        sift_descriptors_by_path[result_path] = sift_desc
                        
                        # Collect SIFT descriptors for KMeans
                        if sift_desc.shape[0] > 0:
                            all_sift_descriptors.append(sift_desc)
                        
                        pbar.set_description(f'exitos: {len(feature_vectors)}')
                        pbar.update(1)
                        processed += 1
                        
                        if args.save_every > 0 and args.out and len(feature_vectors) % args.save_every == 0:
                            _save_partial(args.out, paths, feature_vectors)
                            _safe_print(f"[*] Guardados {len(feature_vectors)} resultados a {args.out}")
            except Exception as loop_exc:
                _safe_print(f"[!] Loop exception at processed={processed}: {loop_exc}")
                import traceback
                _safe_print(traceback.format_exc())
                raise
            finally:
                _safe_print(f"[*] Loop completed: processed={processed}, results={len(feature_vectors)}")

    if not feature_vectors:
        raise RuntimeError("No images could be processed successfully.")
    
    set_tqdm_write(None)  # Reset tqdm.write before final processing
    
    # Fit KMeans on all collected SIFT descriptors
    _safe_print("[*] Starting Bag of Words processing...")
    if all_sift_descriptors:
        all_sift_concatenated = np.vstack(all_sift_descriptors)
        _safe_print(f"[*] Total SIFT descriptors collected: {all_sift_concatenated.shape[0]}")
    else:
        _safe_print("[!] No SIFT descriptors found, creating empty BoW features")
        all_sift_concatenated = np.array([], dtype=np.float32).reshape(0, 128)
    
    # Initialize BoW extractor and fit KMeans
    from src.features.BoW import BoWExtractor
    from src.utils.config import get_config
    n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 100)
    
    bow_extractor = BoWExtractor(n_clusters=n_clusters)
    if all_sift_concatenated.shape[0] > 0:
        bow_extractor.fit(all_sift_concatenated)
    else:
        _safe_print("[!] Warning: No SIFT descriptors to fit KMeans")
    
    # Save the BoW extractor model
    bow_extractor.save(args.bow)
    
    # Create BoW histograms for all images
    _safe_print("[*] Creating Bag of Words histograms...")
    completed_paths = sorted([p for p in paths if p in feature_vectors])
    final_features = []
    
    with tqdm(total=len(completed_paths), unit="imagenes", postfix="BoW") as pbar:
        for path in completed_paths:
            sift_desc = sift_descriptors_by_path[path]
            bow_histogram = bow_extractor.extract_histogram(sift_desc)
            
            # Concatenate original features with BoW histogram
            combined_features = np.concatenate([feature_vectors[path], bow_histogram]).astype(np.float32)
            final_features.append(combined_features)
            pbar.update(1)
    
    # Stack all features
    matrix = np.stack(final_features)
    
    if skipped and _VERBOSE:
        _safe_print(f"[!] Skipped {len(skipped)} images due to read errors.")
    
    # Save final matrix
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, matrix=matrix, paths=[str(p) for p in completed_paths])
    _log_timed(time() - start, "MATRIX", f"{matrix.shape} ({len(completed_paths)} images)")


if __name__ == "__main__":
    main()

