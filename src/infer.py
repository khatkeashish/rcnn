import argparse
import os
from typing import Sequence

import cv2
import numpy as np
import tensorflow as tf
import yaml

from utils import getROI, non_max_suppression_fast, selectiveSearch
from voc2012 import get_labels


def get_model(path: str, input_shape: tuple[int | None, int, int, int]) -> tf.keras.Model:
    model = tf.keras.models.load_model(path)
    model.build(input_shape)
    return model


def get_pred(image: np.ndarray, image_size: int) -> tuple[str, int, float]:
    roi = cv2.resize(image, (image_size, image_size))
    roi = np.asarray(roi, dtype="float32")
    pred = model.predict(roi.reshape((1, image_size, image_size, 3)), verbose=0)

    key_list = list(voc_labels.keys())
    val_list = list(voc_labels.values())

    idx = int(np.argmax(pred))
    prob = float(np.max(pred))
    roi_class = val_list.index(idx)
    roi_label = key_list[roi_class]
    return roi_label, roi_class, prob


class DictList(dict):
    def __setitem__(self, key, value):
        try:
            self[key].append(value)
        except KeyError:
            super().__setitem__(key, [value])
        except AttributeError:
            super().__setitem__(key, [self[key], value])


def load_train_config(config_path: str | None = None) -> dict:
    default_config_path = os.path.join(os.getcwd(), "configs", "train.yaml")
    path = config_path or default_config_path
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def infer_directory(
    input_dir: str,
    model_path: str | None = None,
    run_dir: str | None = None,
    output_dir: str | None = None,
    image_size: int | None = None,
    prob_threshold: float = 0.5,
    overlap_thresh: float = 0.2,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    config = load_train_config()
    resolved_image_size = image_size or int(config.get("image_size", 64))
    backbone_arch = config.get("backbone_arch", "vgg16")

    default_best_keras = os.path.join(
        os.getcwd(),
        "models",
        f"rcnn_{backbone_arch}_voc2012_{resolved_image_size}x{resolved_image_size}_best.keras",
    )
    default_best_h5 = os.path.join(
        os.getcwd(),
        "models",
        f"rcnn_{backbone_arch}_voc2012_{resolved_image_size}x{resolved_image_size}_best.h5",
    )

    resolved_model_path = None
    if model_path:
        resolved_model_path = model_path
    elif run_dir:
        rdir = os.path.abspath(run_dir)
        for candidate in ("best_model.keras", "best_model.h5", "final_model.keras", "final_model.h5"):
            cpath = os.path.join(rdir, candidate)
            if os.path.exists(cpath):
                resolved_model_path = cpath
                break
    else:
        resolved_model_path = default_best_keras if os.path.exists(default_best_keras) else default_best_h5

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Model file not found at {resolved_model_path}. "
            "Pass --model-path explicitly or train the model first."
        )

    global model, voc_labels
    voc_labels = get_labels()
    model = get_model(resolved_model_path, (None, resolved_image_size, resolved_image_size, 3))

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(exts) and os.path.isfile(os.path.join(input_dir, f))
    )
    if not image_files:
        print(f"No images found in {input_dir} with extensions {exts}.")
        return

    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ss_results = selectiveSearch("fast", image)

        predictions = DictList()
        for ss_result in ss_results:
            x1, y1, w, h = ss_result
            ss_box = [x1, y1, x1 + w, y1 + h]
            roi = getROI(image, ss_box)
            if roi is None or roi.size == 0:
                continue
            roi_label, roi_class, prob = get_pred(roi, resolved_image_size)
            if roi_class == voc_labels.get("background", 0):
                continue
            if prob < prob_threshold:
                continue
            predictions[roi_label] = (ss_box, prob)

        output_image = image.copy()
        summary = []
        for label, values in predictions.items():
            boxes = [v[0] for v in values]
            probs = [v[1] for v in values]
            if not boxes:
                continue
            boxes_arr = np.stack(boxes, axis=0)
            probs_arr = np.stack(probs, axis=0)
            kept = non_max_suppression_fast(boxes_arr, probs_arr, overlap_thresh)
            count = len(kept)
            if count == 0:
                continue
            summary.append(f"{label}({count})")
            for box in kept:
                x1, y1, x2, y2 = box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output_image,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        out_name = os.path.splitext(os.path.basename(filename))[0] + "_detections.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        summary_str = ", ".join(summary) if summary else "no objects detected"
        print(f"{filename}: {summary_str} -> {out_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RCNN inference on all images in a directory.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing images to run inference on.",
    )
    parser.add_argument(
        "--model-path",
        required=False,
        help="Path to trained model (.keras/.h5). Defaults to stable best alias derived from train.yaml config.",
    )
    parser.add_argument(
        "--run-dir",
        required=False,
        help="Path to a specific training run directory (uses best_model/final_model inside it).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Directory to write output images with detections overlaid (default: ./inference_outputs).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=False,
        help="Square image size used by the model (default: value from configs/train.yaml or 64).",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Minimum prediction probability to keep a proposal (default: 0.5).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.2,
        help="IoU threshold for non-max suppression (default: 0.2).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.getcwd(), "inference_outputs")

    infer_directory(
        input_dir=input_dir,
        model_path=args.model_path,
        run_dir=args.run_dir,
        output_dir=out_dir,
        image_size=args.image_size,
        prob_threshold=args.prob_threshold,
        overlap_thresh=args.overlap_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
