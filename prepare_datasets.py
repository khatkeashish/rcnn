#!/usr/bin/env python3
"""Prepare both train and test preprocessed datasets and cache them.

Usage:
    python3 prepare_datasets.py --train_data <train_data_dir> --test_data <test_data_dir> [--force]

If `--force` is set the script will re-generate the cache even if present.
"""
import os
import argparse
from preprocessing import preprocess_dataset


def ensure_processed_dir():
    processed_dir = os.path.join(os.getcwd(), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


def build_cache(data_dir, image_shape, voc_labels, out_name, force=False):
    processed_dir = ensure_processed_dir()
    out_path = os.path.join(processed_dir, out_name)
    if os.path.exists(out_path) and not force:
        print(f"Cache exists: {out_path} (use --force to regenerate)")
        return
    print(f"Building cache for {data_dir} -> {out_path}")
    preprocess_dataset(data_dir, image_shape, voc_labels, out_path=out_path)
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=False, help='Path to VOC train/val directory')
    parser.add_argument('--test_data', required=False, help='Path to VOC test directory')
    parser.add_argument('--image_size', type=int, default=64, help='Square image size (default: 64)')
    parser.add_argument('--force', action='store_true', help='Force regeneration of caches')
    parser.add_argument('--out-dir', dest='out_dir', required=False, help='Directory to write processed caches (default: ./processed)')
    args = parser.parse_args()

    # voc_labels from voc2012 module to keep filtering consistent
    from voc2012 import get_labels
    voc_labels = get_labels(filtered=True)

    image_shape = (args.image_size, args.image_size)

    # Choose sensible defaults if not provided (existing folders in repo)
    repo_train_default = os.path.join(os.getcwd(), "VOC2012_train_val", "VOC2012_train_val")
    repo_test_default = os.path.join(os.getcwd(), "VOC2012_test", "VOC2012_test")

    train_data = args.train_data if args.train_data else (repo_train_default if os.path.isdir(repo_train_default) else None)
    test_data = args.test_data if args.test_data else (repo_test_default if os.path.isdir(repo_test_default) else None)

    if not train_data:
        parser.error("Train data path not provided and default train folder not found.")

    # Where to write cache files
    if args.out_dir:
        processed_dir = os.path.abspath(args.out_dir)
        os.makedirs(processed_dir, exist_ok=True)
    else:
        processed_dir = ensure_processed_dir()

    build_cache(train_data, image_shape, voc_labels, os.path.join(processed_dir, "preprocessed_train.pkl"), force=args.force)
    if test_data:
        build_cache(test_data, image_shape, voc_labels, os.path.join(processed_dir, "preprocessed_test.pkl"), force=args.force)

    print("All requested caches are prepared.")
