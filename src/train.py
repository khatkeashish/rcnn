import argparse
import os
import pickle
from collections.abc import Sequence

import tensorflow as tf
import yaml

from models import Backbone, Model
from preprocessing import preprocess_dataset
from utils import Configs
from voc2012 import get_labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        required=False,
        help="Directory to write/read cached preprocessed files (default: ./processed)",
    )
    parser.add_argument(
        "--regen-cache",
        action="store_true",
        help="Force regeneration of the preprocess cache",
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
        help="Base name for cache files (default derived from out-path and image size)",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        required=False,
        help="Process in chunks of this many images to limit memory",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        required=False,
        help="TensorBoard log directory (default: <out-dir>/logs)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        required=False,
        help="Path to YAML config for training (default: ./configs/train.yaml)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    voc_labels = get_labels()

    # Load training configuration from YAML (with sensible defaults)
    default_config_path = os.path.join(os.getcwd(), "configs", "train.yaml")
    config_path = args.config if args.config is not None else default_config_path
    config_data: dict = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_data = loaded

    data_dir = config_data.get("data_dir", "data/VOC2012_train_val/VOC2012_train_val")
    num_classes = len(voc_labels)
    dropout_rate = float(config_data.get("dropout_rate", 0.35))
    learning_rate = float(config_data.get("learning_rate", 0.0001))
    test_size = float(config_data.get("test_size", 0.1))
    image_size = int(config_data.get("image_size", 64))
    image_shape = (image_size, image_size)
    batch_size = int(config_data.get("batch_size", 256))
    epochs = int(config_data.get("epochs", 100))
    optimizer = config_data.get("optimizer", "Adam")
    loss = config_data.get("loss", "categorical_crossentropy")
    metrics = config_data.get("metrics", ["accuracy"])

    # Fraction of total epochs reserved for fine-tuning the backbone at the end of training.
    # Values <= 0.0 disable backbone fine-tuning; values >= 1.0 train the backbone for all epochs.
    backbone_train_fraction = float(config_data.get("backbone_train_fraction", 0.1))

    configs = Configs(
        data_dir,
        num_classes,
        dropout_rate,
        learning_rate,
        test_size,
        image_shape,
        batch_size,
        epochs,
        optimizer,
        loss,
        metrics,
    )

    cache_dir = config_data.get("cache_dir")
    if args.out_dir:
        processed_dir = os.path.abspath(args.out_dir)
        os.makedirs(processed_dir, exist_ok=True)
    elif cache_dir:
        processed_dir = os.path.abspath(cache_dir)
        os.makedirs(processed_dir, exist_ok=True)
    else:
        processed_dir = os.path.join(os.getcwd(), "processed")
        os.makedirs(processed_dir, exist_ok=True)

    cache_path = os.path.join(processed_dir, "preprocessed_train.pkl")
    if args.regen_cache and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass
    workers = getattr(args, "workers", None) or config_data.get("workers")
    cache_name = getattr(args, "cache_name", None) or config_data.get("cache_name")
    chunk_size = getattr(args, "chunk_size", None) or config_data.get("chunk_size")
    X_train, y_train = preprocess_dataset(
        configs.data_dir,
        configs.image_shape,
        voc_labels,
        out_path=cache_path,
        workers=workers,
        cache_name=cache_name,
        chunk_size=chunk_size,
    )
    y_train = tf.keras.utils.to_categorical(y_train, len(voc_labels))

    # Load separate preprocessed test set (used as validation data)
    test_cache_path = os.path.join(processed_dir, "preprocessed_test.pkl")
    if not os.path.exists(test_cache_path):
        msg = (
            f"Validation cache not found at {test_cache_path}. "
            "Run `make prepare` (or src/prepare_datasets.py) to build train/test caches."
        )
        raise FileNotFoundError(msg)

    with open(test_cache_path, "rb") as f:
        test_data = pickle.load(f)
    X_val = test_data.get("X")
    y_val = test_data.get("y")
    if X_val is None or y_val is None:
        raise ValueError(f"Invalid validation cache format in {test_cache_path}: expected keys 'X' and 'y'.")
    y_val = tf.keras.utils.to_categorical(y_val, len(voc_labels))

    print("Training images = ", len(X_train))
    print("Validation images = ", len(X_val))
    print(X_train.shape)
    print(y_train.shape)

    trdata = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, rotation_range=90
    )
    traindata = trdata.flow(x=X_train, y=y_train)
    tsdata = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=False, vertical_flip=False, rotation_range=0
    )
    valdata = tsdata.flow(x=X_val, y=y_val)

    backbone_arch = config_data.get("backbone_arch", "vgg16")
    backbone = Backbone(
        arch=backbone_arch,
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
        trainable=False,
    )
    backbone_model = backbone.backboneModel()
    backbone_model.compile(optimizer, loss, metrics)
    backbone_model.summary()

    model = Model(
        backbone_model=backbone.backboneModel(),
        output_classes=configs.num_classes,
        dropout_rate=configs.dropout_rate,
    )
    model.compile(
        optimizer=configs.optimizer,
        loss=configs.loss,
        metrics=configs.metrics,
    )

    model.summary()

    total_epochs = configs.epochs
    if backbone_train_fraction <= 0.0:
        frozen_epochs = total_epochs
        fine_tune_epochs = 0
    elif backbone_train_fraction >= 1.0:
        frozen_epochs = 0
        fine_tune_epochs = total_epochs
    else:
        fine_tune_epochs = max(1, int(total_epochs * backbone_train_fraction))
        if fine_tune_epochs > total_epochs:
            fine_tune_epochs = total_epochs
        frozen_epochs = total_epochs - fine_tune_epochs

    # Ensure models directory exists and use descriptive filenames (include backbone name)
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, f"rcnn_{backbone_arch}_voc2012_64x64_best.h5")
    final_model_path = os.path.join(models_dir, f"rcnn_{backbone_arch}_voc2012_64x64_final.h5")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        verbose=1,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, verbose=1, mode="auto")

    callbacks = [checkpoint, early]
    if args.tensorboard:
        if args.logdir:
            tb_logdir = os.path.abspath(args.logdir)
        elif "logdir" in config_data:
            tb_logdir = os.path.abspath(config_data["logdir"])
        else:
            tb_logdir = os.path.join(processed_dir, "logs")
        os.makedirs(tb_logdir, exist_ok=True)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logdir,
            histogram_freq=1,
            write_graph=True,
            update_freq="batch",
        )
        callbacks.append(tensorboard_cb)
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir {tb_logdir}")

    # Phase 1: train with frozen backbone (only classification head), if configured.
    if frozen_epochs > 0:
        model.fit(
            traindata,
            batch_size=configs.batch_size,
            epochs=frozen_epochs,
            callbacks=callbacks,
            validation_data=valdata,
            verbose=1,
            shuffle=True,
        )

    # Phase 2: unfreeze backbone and fine-tune for the remaining epochs.
    if fine_tune_epochs > 0:
        # Ensure the backbone inside the composite model is trainable.
        try:
            model.layers[0].trainable = True
        except (AttributeError, IndexError):
            backbone_model.trainable = True

        model.compile(
            optimizer=configs.optimizer,
            loss=configs.loss,
            metrics=configs.metrics,
        )

        model.fit(
            traindata,
            batch_size=configs.batch_size,
            epochs=total_epochs,
            initial_epoch=frozen_epochs,
            callbacks=callbacks,
            validation_data=valdata,
            verbose=1,
            shuffle=True,
        )

    model.save(final_model_path)

    print()
    print()
    print("Training finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
