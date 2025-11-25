import os
import tensorflow as tf
import cv2
import numpy as np

from voc2012 import get_labels, getFilepaths, loadImage, loadAnnotation, plot_annotations
from utils import Configs, selectiveSearch, getROI, non_max_suppression_fast
from preprocessing import preprocess_dataset
import argparse


def get_model(path, input_shape):
    model = tf.keras.models.load_model(path)
    model.build(input_shape)
    return model

def get_pred(image):
    roi = cv2.resize(image, (64, 64)) # resize the image
    roi = np.asarray(roi, dtype='float32')
    pred = model.predict(roi.reshape((1, 64, 64, 3)))
    # list out keys and values separately
    key_list = list(voc_labels.keys())
    val_list = list(voc_labels.values())

    idx = np.argmax(pred)
    prob = np.max(pred)
    roi_class = val_list.index(idx)
    roi_label = key_list[roi_class]
    return roi_label, roi_class, prob

class DictList(dict):
    def __init__(self):
        super().__init__()
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value)
        except KeyError: # If it fails, because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError: # If it fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Run evaluation on preprocessed dataset')
    parser.add_argument('--out-dir', dest='out_dir', required=False, help='Directory to read/write cached preprocessed files (default: ./processed)')
    parser.add_argument('--regen-cache', action='store_true', help='Force regeneration of the preprocess cache')
    args = parser.parse_args()
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # get trainewd labels
    voc_labels = get_labels(filtered=True)

    
    # Create arguments
    data_dir = "/home/ash/Documents/Personal/Projects/odts/benchmarks/VOC2012"
    # data_dir = "/home/ash/Documents/Projects/VOC2012"
    num_classes = len(voc_labels)
    dropout_rate = 0.35
    learning_rate = 0.0001
    test_size = 0.1
    drop_rate = 0.5
    image_shape = (64,64)
    batch_size = 32
    epochs = 100
    input_shape = (image_shape[0], image_shape[1], 3)
    optimizer="Adam"
    loss="categorical_crossentropy"
    metrics=["accuracy"]

    configs = Configs(data_dir, num_classes, dropout_rate, learning_rate, test_size, image_shape, batch_size, epochs, optimizer, loss, metrics)
    #Load the pre-trained model
    model = get_model("person_model", (None, 64, 64, 3))
    model.summary()

    if args.out_dir:
        processed_dir = os.path.abspath(args.out_dir)
        os.makedirs(processed_dir, exist_ok=True)
    else:
        processed_dir = os.path.join(os.getcwd(), "processed")
        os.makedirs(processed_dir, exist_ok=True)
    cache_path = os.path.join(processed_dir, "preprocessed_test.pkl")
    if args.eval:
        # If regen requested, remove existing cache first
        if args.regen_cache and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
        # Load preprocessed dataset (will use cache if exists)
        X, y = preprocess_dataset(configs.data_dir, configs.image_shape, voc_labels, out_path=cache_path)
        y_cat = tf.keras.utils.to_categorical(y, len(voc_labels))
        loss, acc = model.evaluate(X, y_cat, batch_size=32)
        print(f"Eval loss={loss:.4f} acc={acc:.4f}")
        exit(0)

    # Default behavior: run selective search on a single image and show detections
    image_paths, annotation_paths = getFilepaths(configs.data_dir)
    idx = 1
    image = loadImage(image_paths[idx])
    annotation = loadAnnotation(annotation_paths[idx])
    print(annotation)

    # Selective search on the image
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss_results = selectiveSearch("fast", image)

    predictions = DictList()
    for ss_result in ss_results:
        x1, y1, w, h = ss_result
        ss_box = [x1, y1, x1 + w, y1 + h]

        # Get the roi of the box
        roi = getROI(image, ss_box)  # crop the image
        roi_label, roi_class, prob = get_pred(roi)
        if roi_class:
            predictions[roi_label] = ss_box, prob

    detections = []
    keys = list(predictions.keys())
    for key in keys:
        values = predictions[key]
        boxes = []
        probs = []
        for i in range(len(values)):
            boxes.append(values[i][0])
            probs.append(values[i][1])
        overlapThresh = 0.2
        boxes = np.stack(boxes, axis=0)
        probs = np.stack(probs, axis=0)
        non_max_boxes = non_max_suppression_fast(boxes, probs, overlapThresh)
        detections.append([key, non_max_boxes])
        print(detections)
        plot_annotations(image, detections=detections, ground_truth=annotation)








