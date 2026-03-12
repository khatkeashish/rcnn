# Makefile for dataset preparation, training, and tooling

.PHONY: prepare prepare-force train train-force tensorboard format


prepare:
	@echo "Preparing datasets (train/test caches)"
	@uv run python src/prepare_datasets.py --config configs/prepare.yaml

prepare-force:
	@echo "Preparing datasets (force regenerate caches)"
	@uv run python src/prepare_datasets.py --config configs/prepare.yaml --force


train:
	@echo "Training RCNN model"
	@uv run python src/train.py --config configs/train.yaml --tensorboard

train-force:
	@echo "Training RCNN model (regenerating cache)"
	@uv run python src/train.py --config configs/train.yaml --tensorboard --regen-cache

tensorboard:
	@echo "Starting TensorBoard on logs/"
	@tensorboard --logdir logs

format:
	@echo "Formatting and linting Python sources"
	@uv run ruff check src --fix --exit-zero && uv run ruff format src
