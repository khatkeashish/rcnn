# Makefile for dataset preparation

.PHONY: prepare prepare-force

# Default dataset paths inside the repo (these are the folders used by the repo)
TRAIN_DIR := VOC2012_train_val/VOC2012_train_val
TEST_DIR := VOC2012_test/VOC2012_test
OUT_DIR ?= processed

prepare:
	python3 prepare_datasets.py --train_data $(TRAIN_DIR) --test_data $(TEST_DIR) --out-dir $(OUT_DIR)

prepare-force:
	python3 prepare_datasets.py --train_data $(TRAIN_DIR) --test_data $(TEST_DIR) --out-dir $(OUT_DIR) --force
