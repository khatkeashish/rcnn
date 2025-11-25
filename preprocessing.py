import os
import numpy as np
from tqdm import tqdm
import pickle

from voc2012 import getFilepaths, loadImage, loadAnnotation
from utils import selectiveSearch, getROI, bb_intersection_over_union
from multiprocessing import cpu_count, Pool
from functools import partial


def roi_extractor(image, annotation, image_shape, voc_labels):
    X = []
    y = []
    img_classes = []
    img_boxes = []

    for obj in annotation["objects"]:
        obj_class = list(obj.keys())[0]
        if obj_class in voc_labels:
            obj_box = list(obj.values())[0]
            img_classes.append(obj_class)
            img_boxes.append(obj_box)

    if not img_classes:
        return X, y

    max_background_images = 5 * len(img_classes)
    num_background_images = 0

    ss_results = selectiveSearch("fast", image)
    background_images = []
    for ss_result in ss_results:
        x1, y1, w, h = ss_result
        ss_box = [x1, y1, x1 + w, y1 + h]

        iou_list = []
        for idx in range(len(img_classes)):
            iou = bb_intersection_over_union(ss_box, img_boxes[idx])
            iou_list.append(iou)

        if not iou_list:
            continue

        iou_max = max(iou_list)
        if iou_max > 0.7:
            roi = getROI(image, ss_box)
            roi = __resize_roi(roi, image_shape)
            roi_class = img_classes[iou_list.index(iou_max)]
            roi_label = voc_labels.get(roi_class)
            X.append(roi)
            y.append(roi_label)
        elif iou_max < 0.2 and num_background_images < max_background_images:
            roi = getROI(image, ss_box)
            roi = __resize_roi(roi, image_shape)
            background_images.append(roi)
            num_background_images += 1

    background_labels = [0] * len(background_images)
    X.extend(background_images)
    y.extend(background_labels)
    return X, y


def __resize_roi(roi, image_shape):
    import cv2
    if roi is None or roi.size == 0:
        return np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    try:
        roi = cv2.resize(roi, (image_shape[0], image_shape[1]))
    except Exception:
        roi = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    return roi


def _process_single(idx, image_paths, annotation_paths, image_shape, voc_labels):
    # Helper to process one image/annotation pair
    image = loadImage(image_paths[idx])
    annotation = loadAnnotation(annotation_paths[idx])
    X, y = roi_extractor(image, annotation, image_shape, voc_labels)
    return X, y


def create_dataset(image_paths, annotation_paths, image_shape, voc_labels, workers=None):
    images = []
    labels = []
    n = len(image_paths)
    if workers is None:
        workers = max(1, int(0.8 * cpu_count()))

    if workers == 1:
        # Serial fallback
        for idx in tqdm(range(n), desc="Creating dataset"):
            X, y = _process_single(idx, image_paths, annotation_paths, image_shape, voc_labels)
            images.extend(X)
            labels.extend(y)
    else:
        with Pool(processes=workers) as pool:
            func = partial(_process_single, image_paths=image_paths, annotation_paths=annotation_paths, image_shape=image_shape, voc_labels=voc_labels)
            # imap to keep order and use tqdm for progress
            for X, y in tqdm(pool.imap(func, range(n)), total=n, desc="Creating dataset"):
                images.extend(X)
                labels.extend(y)

    return np.array(images), np.array(labels)


def preprocess_dataset(data_dir, image_shape, voc_labels, out_path=None, workers=None):
    """Preprocess dataset and cache to `out_path` using pickle.
    If cache exists it will be loaded instead of re-running processing.
    Returns: (X, y) numpy arrays
    """
    # Default output path placed under repo-level `processed/` directory
    if out_path is None:
        processed_dir = os.path.join(os.getcwd(), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        out_path = os.path.join(processed_dir, "preprocessed.pkl")
    else:
        # Ensure containing directory exists
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if os.path.exists(out_path):
        try:
            with open(out_path, "rb") as f:
                data = pickle.load(f)
            return data.get("X"), data.get("y")
        except Exception:
            # fallthrough to regenerate
            pass

    image_paths, annotation_paths = getFilepaths(data_dir)
    X, y = create_dataset(image_paths, annotation_paths, image_shape, voc_labels, workers=workers)
    try:
        with open(out_path, "wb") as f:
            pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # If saving fails, ignore but return the data
        pass
    return X, y
