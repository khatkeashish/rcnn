import os
import numpy as np
from tqdm import tqdm

# Local imports
from voc2012 import getFilepaths, loadImage, loadAnnotation
from utils import selectiveSearch, getROI, bb_intersection_over_union


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


def create_dataset(image_paths, annotation_paths, image_shape, voc_labels):
    images = []
    labels = []
    for idx in tqdm(range(len(image_paths)), desc="Creating dataset"):
        image = loadImage(image_paths[idx])
        annotation = loadAnnotation(annotation_paths[idx])
        X, y = roi_extractor(image, annotation, image_shape, voc_labels)
        images.extend(X)
        labels.extend(y)
    return np.array(images), np.array(labels)


def preprocess_dataset(data_dir, image_shape, voc_labels, out_path="preprocessed.npz"):
    if os.path.exists(out_path):
        npz = np.load(out_path)
        return npz["X"], npz["y"]

    image_paths, annotation_paths = getFilepaths(data_dir)
    X, y = create_dataset(image_paths, annotation_paths, image_shape, voc_labels)
    np.savez_compressed(out_path, X=X, y=y)
    return X, y
