#!/usr/bin/env python3
"""Prepare both train and test preprocessed datasets and cache them."""

import argparse
import os
from collections.abc import Sequence

from preprocessing import preprocess_dataset
from voc2012 import get_labels


def ensure_processed_dir(base_dir: str | None = None) -> str:
    root = base_dir if base_dir is not None else os.getcwd()
    processed_dir = os.path.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


def build_cache(
    data_dir,
    image_shape,
    voc_labels,
    out_path,
    force: bool = False,
    workers=None,
    cache_name=None,
    chunk_size=None,
) -> None:
    if os.path.exists(out_path) and not force:
        print(f"Cache exists: {out_path} (use --force to regenerate)")
        return
    print(f"Building cache for {data_dir} -> {out_path}")
    preprocess_dataset(
        data_dir,
        image_shape,
        voc_labels,
        out_path=out_path,
        workers=workers,
        cache_name=cache_name,
        chunk_size=chunk_size,
    )
    print("Done")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=False, help="Path to VOC train/val directory")
    parser.add_argument("--test_data", required=False, help="Path to VOC test directory")
    parser.add_argument("--image_size", type=int, default=64, help="Square image size (default: 64)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of caches")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        required=False,
        help="Directory to write processed caches (default: ./processed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        help="Number of worker processes to use (default: 80% of CPUs)",
    )
    parser.add_argument(
        "--cache-name",
        dest="cache_name",
        required=False,
        help="Base name for cache files (default derived from out-path)",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        required=False,
        help="Process dataset in chunks of this many images to limit memory (default: disabled)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    voc_labels = get_labels()

    image_shape = (args.image_size, args.image_size)

    repo_train_default = os.path.join(os.getcwd(), "data", "VOC2012_train_val", "VOC2012_train_val")
    repo_test_default = os.path.join(os.getcwd(), "data", "VOC2012_test", "VOC2012_test")

    train_data = (
        args.train_data if args.train_data else (repo_train_default if os.path.isdir(repo_train_default) else None)
    )
    test_data = args.test_data if args.test_data else (repo_test_default if os.path.isdir(repo_test_default) else None)

    if not train_data:
        parser.error("Train data path not provided and default train folder not found.")

    if args.out_dir:
        processed_dir = os.path.abspath(args.out_dir)
        os.makedirs(processed_dir, exist_ok=True)
    else:
        processed_dir = ensure_processed_dir()

    workers = args.workers
    train_out = os.path.join(processed_dir, "preprocessed_train.pkl")
    build_cache(
        train_data,
        image_shape,
        voc_labels,
        train_out,
        force=args.force,
        workers=workers,
        cache_name=args.cache_name,
        chunk_size=args.chunk_size,
    )

    test_out = os.path.join(processed_dir, "preprocessed_test.pkl")
    build_cache(
        test_data,
        image_shape,
        voc_labels,
        test_out,
        force=args.force,
        workers=workers,
        cache_name=args.cache_name,
        chunk_size=args.chunk_size,
    )

    print("All requested caches are prepared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
