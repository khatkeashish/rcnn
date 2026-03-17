import argparse
import json
import os
import pickle
import platform
from collections.abc import Sequence
from datetime import datetime, timezone

import tensorflow as tf
import yaml

from models import Backbone, Model
from preprocessing import preprocess_dataset
from utils import Configs
from voc2012 import get_labels


def _load_yaml_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _safe_token(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(s))


def _build_run_id(effective_config: dict) -> str:
    backbone = _safe_token(effective_config.get("backbone_arch", "vgg16"))
    image_size = int(effective_config.get("image_size", 64))
    batch_size = int(effective_config.get("batch_size", 256))
    lr = effective_config.get("learning_rate", 0.0001)
    run_name = effective_config.get("run_name") or effective_config.get("experiment_name") or ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [f"{backbone}", f"s{image_size}", f"bs{batch_size}", f"lr{lr}", ts]
    if run_name:
        parts.insert(0, _safe_token(run_name))
    return "_".join(parts)


def _resolve_optimizer(optimizer_spec, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    if isinstance(optimizer_spec, tf.keras.optimizers.Optimizer):
        optimizer_spec.learning_rate = learning_rate
        return optimizer_spec
    if isinstance(optimizer_spec, dict):
        cfg = dict(optimizer_spec)
        cfg.setdefault("config", {})
        if isinstance(cfg["config"], dict):
            cfg["config"]["learning_rate"] = learning_rate
        return tf.keras.optimizers.get(cfg)
    return tf.keras.optimizers.get({"class_name": str(optimizer_spec), "config": {"learning_rate": learning_rate}})


def _resolve_resume_path(resume_from: str, run_dir: str | None = None) -> str | None:
    if not resume_from:
        return None
    path = os.path.abspath(resume_from)
    if os.path.isdir(path):
        for candidate in ("best_model.keras", "final_model.keras", "best_model.h5", "final_model.h5"):
            cpath = os.path.join(path, candidate)
            if os.path.exists(cpath):
                return cpath
        ckpt_dir = os.path.join(path, "checkpoints")
        if os.path.isdir(ckpt_dir):
            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                return max(files, key=os.path.getmtime)
        return None
    if os.path.exists(path):
        return path
    if run_dir:
        alt = os.path.join(run_dir, resume_from)
        if os.path.exists(alt):
            return alt
    return None


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
    parser.add_argument(
        "--run-name",
        dest="run_name",
        required=False,
        help="Optional name prefix for the training run (used in output directory naming).",
    )
    parser.add_argument(
        "--models-dir",
        dest="models_dir",
        required=False,
        help="Base directory to write per-run model artifacts (default: ./models/runs).",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        required=False,
        help="Resume training from a checkpoint/model path or a prior run directory.",
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
    config_data: dict = _load_yaml_config(config_path)

    # Merge into an effective config (YAML base + CLI overrides)
    effective_config = dict(config_data)
    if args.run_name:
        effective_config["run_name"] = args.run_name

    data_dir = effective_config.get("data_dir", "data/VOC2012_train_val/VOC2012_train_val")
    num_classes = len(voc_labels)
    dropout_rate = float(effective_config.get("dropout_rate", 0.35))
    learning_rate = float(effective_config.get("learning_rate", 0.0001))
    test_size = float(effective_config.get("test_size", 0.1))
    image_size = int(effective_config.get("image_size", 64))
    image_shape = (image_size, image_size)
    batch_size = int(effective_config.get("batch_size", 256))
    epochs = int(effective_config.get("epochs", 100))
    optimizer = effective_config.get("optimizer", "Adam")
    loss = effective_config.get("loss", "categorical_crossentropy")
    metrics = effective_config.get("metrics", ["accuracy"])

    # Fraction of total epochs reserved for fine-tuning the backbone at the end of training.
    # Values <= 0.0 disable backbone fine-tuning; values >= 1.0 train the backbone for all epochs.
    backbone_train_fraction = float(effective_config.get("backbone_train_fraction", 0.1))

    # Reproducibility (best-effort)
    seed = int(effective_config.get("seed", 1337))
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

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

    cache_dir = effective_config.get("cache_dir")
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
    workers = getattr(args, "workers", None) or effective_config.get("workers")
    cache_name = getattr(args, "cache_name", None) or effective_config.get("cache_name")
    chunk_size = getattr(args, "chunk_size", None) or effective_config.get("chunk_size")

    # Per-run model artifact directory
    models_base_dir = (
        os.path.abspath(args.models_dir) if args.models_dir else os.path.join(os.getcwd(), "models", "runs")
    )
    os.makedirs(models_base_dir, exist_ok=True)
    run_id = _build_run_id(
        {
            **effective_config,
            "backbone_arch": effective_config.get("backbone_arch", "vgg16"),
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    )
    run_dir = os.path.join(models_base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Persist effective config and run metadata
    effective_config = {
        **effective_config,
        "resolved": {
            "config_path": os.path.abspath(config_path),
            "processed_dir": processed_dir,
            "train_cache_path": cache_path,
            "val_cache_path": os.path.join(processed_dir, "preprocessed_test.pkl"),
            "run_id": run_id,
            "run_dir": run_dir,
            "models_base_dir": models_base_dir,
            "seed": seed,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
    }
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(effective_config, f, sort_keys=False)
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
    traindata = trdata.flow(x=X_train, y=y_train, batch_size=configs.batch_size, shuffle=True)
    tsdata = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=False, vertical_flip=False, rotation_range=0
    )
    valdata = tsdata.flow(x=X_val, y=y_val, batch_size=configs.batch_size, shuffle=False)

    backbone_arch = effective_config.get("backbone_arch", "vgg16")
    backbone = Backbone(
        arch=backbone_arch,
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
        trainable=False,
    )
    backbone_model = backbone.backboneModel()
    resolved_optimizer = _resolve_optimizer(optimizer, learning_rate)
    backbone_model.compile(resolved_optimizer, loss, metrics)
    backbone_model.summary()

    model = Model(
        backbone_model=backbone.backboneModel(),
        output_classes=configs.num_classes,
        dropout_rate=configs.dropout_rate,
    )
    model.compile(
        optimizer=resolved_optimizer,
        loss=configs.loss,
        metrics=configs.metrics,
    )

    model.summary()

    resume_from = args.resume_from or effective_config.get("resume_from")
    resume_path = _resolve_resume_path(str(resume_from), run_dir=run_dir) if resume_from else None
    if resume_path:
        try:
            print(f"Resuming weights from: {resume_path}")
            model.load_weights(resume_path)
        except Exception as e:
            print(f"Failed to resume from {resume_path}: {e}")

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

    # Per-run model paths
    best_model_path = os.path.join(run_dir, "best_model.keras")
    final_model_path = os.path.join(run_dir, "final_model.keras")

    early_cfg = effective_config.get("early_stopping", {})
    if not isinstance(early_cfg, dict):
        early_cfg = {}
    early_enabled = bool(early_cfg.get("enabled", True))
    early_monitor = str(early_cfg.get("monitor", "val_loss"))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_patience = int(early_cfg.get("patience", 5))
    early_mode = str(early_cfg.get("mode", "auto"))
    early_restore_best_weights = bool(early_cfg.get("restore_best_weights", False))

    checkpoint_monitor = str(effective_config.get("checkpoint_monitor", early_monitor))
    checkpoint_mode = str(effective_config.get("checkpoint_mode", "auto"))

    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        verbose=1,
        monitor=checkpoint_monitor,
        save_best_only=True,
        save_weights_only=False,
        mode=checkpoint_mode,
    )

    monitor_token = _safe_token(checkpoint_monitor)
    if checkpoint_monitor.startswith("val_"):
        epoch_pattern = os.path.join(
            ckpt_dir, f"weights_epoch{{epoch:03d}}_{monitor_token}={{val_loss:.4f}}.weights.h5"
        )
    else:
        epoch_pattern = os.path.join(ckpt_dir, f"weights_epoch{{epoch:03d}}_{monitor_token}={{loss:.4f}}.weights.h5")
    epoch_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        epoch_pattern,
        verbose=0,
        monitor=checkpoint_monitor,
        save_best_only=False,
        save_weights_only=True,
        mode=checkpoint_mode,
        save_freq="epoch",
    )

    callbacks = [best_checkpoint, epoch_checkpoint]
    if early_enabled and early_patience > 0:
        early = tf.keras.callbacks.EarlyStopping(
            monitor=early_monitor,
            min_delta=early_min_delta,
            patience=early_patience,
            verbose=1,
            mode=early_mode,
            restore_best_weights=early_restore_best_weights,
        )
        callbacks.append(early)

    if args.tensorboard:
        if args.logdir:
            tb_logdir = os.path.abspath(args.logdir)
        elif "logdir" in effective_config:
            tb_logdir = os.path.abspath(effective_config["logdir"])
        else:
            tb_logdir = os.path.join(run_dir, "logs")
        os.makedirs(tb_logdir, exist_ok=True)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logdir,
            histogram_freq=1,
            write_graph=True,
            update_freq="batch",
        )
        callbacks.append(tensorboard_cb)
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir {tb_logdir}")

    # Lightweight run log
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"start_time_utc: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"run_id: {run_id}\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"config_path: {os.path.abspath(config_path)}\n")
        f.write(f"train_cache: {cache_path}\n")
        f.write(f"val_cache: {test_cache_path}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"tf: {tf.__version__}\n")
        f.write(f"backbone: {backbone_arch}\n")
        f.write(f"image_size: {image_size}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"epochs: {epochs}\n")
        if resume_path:
            f.write(f"resume_from: {resume_path}\n")
        if args.tensorboard:
            f.write(f"tensorboard_logdir: {tb_logdir}\n")
        f.write("\n")

    # Phase 1: train with frozen backbone (only classification head), if configured.
    full_history: dict[str, list] = {}
    if frozen_epochs > 0:
        hist1 = model.fit(
            traindata,
            epochs=frozen_epochs,
            callbacks=callbacks,
            validation_data=valdata,
            verbose=1,
        )
        for k, v in (hist1.history or {}).items():
            full_history.setdefault(k, []).extend(list(v))

    # Phase 2: unfreeze backbone and fine-tune for the remaining epochs.
    if fine_tune_epochs > 0:
        # Ensure the backbone inside the composite model is trainable.
        try:
            model.layers[0].trainable = True
        except (AttributeError, IndexError):
            backbone_model.trainable = True

        ft_lr = float(effective_config.get("fine_tune_learning_rate", learning_rate))
        ft_optimizer = _resolve_optimizer(optimizer, ft_lr)
        model.compile(
            optimizer=ft_optimizer,
            loss=configs.loss,
            metrics=configs.metrics,
        )

        hist2 = model.fit(
            traindata,
            epochs=total_epochs,
            initial_epoch=frozen_epochs,
            callbacks=callbacks,
            validation_data=valdata,
            verbose=1,
        )
        for k, v in (hist2.history or {}).items():
            full_history.setdefault(k, []).extend(list(v))

    model.save(final_model_path)

    # Stable aliases for inference convenience (latest run per backbone+size)
    models_root = os.path.join(os.getcwd(), "models")
    os.makedirs(models_root, exist_ok=True)
    stable_best = os.path.join(models_root, f"rcnn_{backbone_arch}_voc2012_{image_size}x{image_size}_best.keras")
    stable_final = os.path.join(models_root, f"rcnn_{backbone_arch}_voc2012_{image_size}x{image_size}_final.keras")
    try:
        if os.path.exists(best_model_path):
            tf.keras.models.load_model(best_model_path).save(stable_best)
        tf.keras.models.load_model(final_model_path).save(stable_final)
    except Exception:
        pass

    # Save combined training history
    try:
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2)
    except Exception:
        pass

    # Append a short summary to the run log (best-effort)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"end_time_utc: {datetime.now(timezone.utc).isoformat()}\n")
            best_val_loss = None
            if "val_loss" in full_history and full_history["val_loss"]:
                best_val_loss = float(min(full_history["val_loss"]))
            if best_val_loss is not None:
                f.write(f"best_val_loss: {best_val_loss}\n")
            if "val_accuracy" in full_history and full_history["val_accuracy"]:
                f.write(f"best_val_accuracy: {float(max(full_history['val_accuracy']))}\n")
            f.write("\n")
    except Exception:
        pass

    print()
    print()
    print("Training finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
