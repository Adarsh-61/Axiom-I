import argparse
import math
import time
from pathlib import Path

import cv2

from app.ml.pipeline import extract_feature_vector_from_image, _run_no_face_fallback
from app.ml.evolution import (
    _SEED_CACHE_FILE,
    _SEED_CACHE_VERSION,
    _MODEL_CACHE,
    _MODEL_CACHE_LOCK,
    _cached_model,
    _coerce_feature_vector,
    _save_json,
)


def _collect_files(dataset_root: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for label, y in (("Real", 0), ("Fake", 1)):
        class_dir = dataset_root / label
        if not class_dir.exists():
            print(f"WARN missing class folder: {class_dir}", flush=True)
            continue
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file()]
        items.extend((p, y) for p in files)
    return items


def _eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "inf"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="../test")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--mode", choices=["full", "fallback"], default="full")
    args = parser.parse_args()

    dataset_root = (Path.cwd() / args.dataset_root).resolve()
    files = _collect_files(dataset_root)
    if args.max_images and args.max_images > 0:
        files = files[: args.max_images]

    total = len(files)
    if total == 0:
        print(f"ERROR no files found in {dataset_root}", flush=True)
        return 1

    print(f"START dataset_root={dataset_root} total_files={total}", flush=True)

    start = time.time()
    valid = 0
    skipped = 0
    samples: list[tuple[list[float], int]] = []

    for idx, (image_path, y) in enumerate(files, start=1):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            skipped += 1
        else:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if args.mode == "fallback":
                    result = _run_no_face_fallback(img_rgb, include_steps=False)
                else:
                    result = extract_feature_vector_from_image(img_rgb, include_steps=False)
                feature_vector = _coerce_feature_vector(result.get("feature_vector"))
                if feature_vector is None:
                    skipped += 1
                else:
                    samples.append((feature_vector, y))
                    valid += 1
            except Exception as ex:
                skipped += 1
                if skipped <= 20:
                    print(f"WARN file={image_path.name} reason={ex}", flush=True)

        if idx % args.progress_every == 0 or idx == total:
            elapsed = max(1e-6, time.time() - start)
            rate = idx / elapsed
            rem = total - idx
            eta = rem / rate if rate > 0 else math.inf
            print(
                "PROGRESS "
                f"processed={idx}/{total} "
                f"valid={valid} skipped={skipped} "
                f"mode={args.mode} "
                f"elapsed_sec={elapsed:.1f} rate={rate:.2f}_img_per_sec "
                f"eta={_eta(eta)}",
                flush=True,
            )

    _save_json(
        _SEED_CACHE_FILE,
        {
            "version": _SEED_CACHE_VERSION,
            "samples": samples,
        },
    )
    print(f"SEED_CACHE_SAVED path={_SEED_CACHE_FILE} sample_count={len(samples)}", flush=True)

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE["feedback_mtime"] = None
        _MODEL_CACHE["seed_mtime"] = None
        _MODEL_CACHE["model"] = None

    model, meta = _cached_model()
    print(f"MODEL_REBUILT model_ready={model is not None} meta={meta}", flush=True)
    print("TRAINING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
