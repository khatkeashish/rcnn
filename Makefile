# Makefile for dataset preparation, training, and tooling

.PHONY: prepare prepare-force train train-force tensorboard format

# Default dataset paths inside the repo (these are the folders used by the repo)
TRAIN_DIR := data/VOC2012_train_val/VOC2012_train_val
TEST_DIR := data/VOC2012_test/VOC2012_test
OUT_DIR ?= data/processed
WORKERS ?=
CACHE_NAME ?=
CHUNK_SIZE ?=
LOGDIR ?= logs
REGEN_CACHE ?=
FORMAT_PATH ?= src
RUFF_CHECK_ARGS ?=
RUFF_FORMAT_ARGS ?=


prepare:
	@CMD="uv run python src/prepare_datasets.py --train_data $(TRAIN_DIR) --test_data $(TEST_DIR) --out-dir $(OUT_DIR)"; \
	if [ -n "$(WORKERS)" ]; then CMD="$$CMD --workers $(WORKERS)"; fi; \
	if [ -n "$(CACHE_NAME)" ]; then CMD="$$CMD --cache-name $(CACHE_NAME)"; fi; \
	if [ -n "$(CHUNK_SIZE)" ]; then CMD="$$CMD --chunk-size $(CHUNK_SIZE)"; fi; \
	echo "Running: $$CMD"; \
	eval $$CMD

prepare-force:
	@CMD="uv run python src/prepare_datasets.py --train_data $(TRAIN_DIR) --test_data $(TEST_DIR) --out-dir $(OUT_DIR) --force"; \
	if [ -n "$(WORKERS)" ]; then CMD="$$CMD --workers $(WORKERS)"; fi; \
	if [ -n "$(CACHE_NAME)" ]; then CMD="$$CMD --cache-name $(CACHE_NAME)"; fi; \
	if [ -n "$(CHUNK_SIZE)" ]; then CMD="$$CMD --chunk-size $(CHUNK_SIZE)"; fi; \
	echo "Running: $$CMD"; \
	eval $$CMD


train:
	@CMD="uv run python src/train.py --out-dir $(OUT_DIR) --tensorboard --logdir $(LOGDIR)"; \
	if [ -n "$(WORKERS)" ]; then CMD="$$CMD --workers $(WORKERS)"; fi; \
	if [ -n "$(CACHE_NAME)" ]; then CMD="$$CMD --cache-name $(CACHE_NAME)"; fi; \
	if [ -n "$(CHUNK_SIZE)" ]; then CMD="$$CMD --chunk-size $(CHUNK_SIZE)"; fi; \
	echo "Running: $$CMD"; \
	eval $$CMD

train-force:
	@CMD="uv run python src/train.py --out-dir $(OUT_DIR) --tensorboard --logdir $(LOGDIR) --regen-cache"; \
	if [ -n "$(WORKERS)" ]; then CMD="$$CMD --workers $(WORKERS)"; fi; \
	if [ -n "$(CACHE_NAME)" ]; then CMD="$$CMD --cache-name $(CACHE_NAME)"; fi; \
	if [ -n "$(CHUNK_SIZE)" ]; then CMD="$$CMD --chunk-size $(CHUNK_SIZE)"; fi; \
	echo "Running: $$CMD"; \
	eval $$CMD

tensorboard:
	@CMD="tensorboard --logdir $(LOGDIR)"; \
	echo "Running: $$CMD"; \
	eval $$CMD

format:
	@CMD="uv run ruff check $(FORMAT_PATH) --fix --exit-zero"; \
	if [ -n "$(RUFF_CHECK_ARGS)" ]; then CMD="$$CMD $(RUFF_CHECK_ARGS)"; fi; \
	CMD="$$CMD && uv run ruff format $(FORMAT_PATH)"; \
	if [ -n "$(RUFF_FORMAT_ARGS)" ]; then CMD="$$CMD $(RUFF_FORMAT_ARGS)"; fi; \
	echo "Running: $$CMD"; \
	eval $$CMD
