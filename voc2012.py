import numpy as np
import os
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from tqdm import tqdm


def get_labels(filtered):
	if filtered:
		# filter the classes for training
		voc_labels = {
    		"background": 0,
    		"person": 1
			}
	else:
				
		# voc dataset classes 
		voc_labels = {
			"background": 0,
			"aeroplane": 1,
			"bicycle": 2,
			"bird": 3,
			"boat": 4,
			"bottle": 5,
			"bus": 6,
			"car": 7,
			"cat": 8,
			"chair": 9,
			"cow": 10,
			"dining_table": 11,
			"dog": 12,
			"horse": 13,
			"motorbike": 14,
			"person": 15,
			"potted_plant": 16,
			"sheep": 17,
			"sofa":18,
			"train":19,
			"tvmonitor": 20
		}
	return voc_labels

def getFilepaths(DATA_DIR):
    print("Reading paths for images and annotations")

    ANNOTATION_DIR = os.path.join(DATA_DIR, "Annotations")
    IMAGES_DIR = os.path.join(DATA_DIR, "JPEGImages")

    annotation_filenames = sorted(os.listdir(ANNOTATION_DIR))
    image_paths = []
    annotation_paths = []
    for num, filename in enumerate(annotation_filenames):
        annotation_path = os.path.join(ANNOTATION_DIR, filename)
        image_filename = filename.split(".")[0]
        image_path = os.path.join(IMAGES_DIR, image_filename + ".jpg")
        image_paths.append(image_path)
        annotation_paths.append(annotation_path)
    assert(len(image_paths) == len(annotation_filenames))
    return image_paths, annotation_paths

# Parse the xml annotation file and retrieve the path to each image, its size and annotations
def extract_xml_annotation(filename):
    """Parse the xml file
    :param filename: str
    """
    z = etree.parse(filename)
    objects = z.findall("./object")
    size = (int(float(z.find(".//width").text)), int(float(z.find(".//height").text)))
    fname = z.find("./filename").text
    dicts = [
        {
            obj.find("name").text: [
                int(float(obj.find("bndbox/xmin").text)),
                int(float(obj.find("bndbox/ymin").text)),
                int(float(obj.find("bndbox/xmax").text)),
                int(float(obj.find("bndbox/ymax").text)),
            ]
        }
        for obj in objects
    ]
    return {"size": size, "filename": fname, "objects": dicts}


def loadAnnotation(filename):
    annotation = extract_xml_annotation(filename)
    return annotation


def loadImage(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def loadData(DATA_DIR):
    print("Load data")

    ANNOTATION_DIR = os.path.join(DATA_DIR, "Annotations")
    IMAGES_DIR = os.path.join(DATA_DIR, "JPEGImages")

    annotation_names = sorted(os.listdir(ANNOTATION_DIR))
    annotations = []
    images = []

    for num, filename in enumerate(tqdm(annotation_names)):
        annotation_path = os.path.join(ANNOTATION_DIR, filename)
        annotation = loadAnnotation(annotation_path)
        image_filename = filename.split(".")[0]
        image_path = os.path.join(IMAGES_DIR, image_filename + ".jpg")
        image = loadImage(image_path)
        annotations.append(annotation)
        images.append(image)
        #if num==0:
         #   break

    print("Loaded images = ", len(images))
    return images, annotations



def patch(axis, bbox, display_txt, color):
    coords = (bbox[0], bbox[1]), bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    axis.text(
        bbox[0],
        bbox[1],
        display_txt,
        color="white",
        bbox={"facecolor": color, "alpha": 0.5},
    )


def plot_annotations(img, detections=None, ground_truth=None):
    current_axis = plt.gca()
    for object in ground_truth["objects"]:
        obj_class = list(object.keys())[0]
        bbox = list(object.values())[0]
        if ground_truth:
            text = "GT " + obj_class
            patch(current_axis, bbox, text, "red")
        if detections:
            for obj_class, boxes in detections:
                for box in boxes:
                    text = "PRED " + obj_class
                    patch(current_axis, box, text, "blue")
        plt.axis("off")
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    DATA_DIR = "/home/ash/Documents/Personal/Projects/odts/benchmarks/VOC2012"
    X, y = loadData(DATA_DIR)

    # Show one image with ground thruth annotation 
    idx = 0  # random.randint(0, len(images))
    image = X[idx]
    annotation = y[idx]
    print(annotation)
    plot_annotations(image, ground_truth=annotation)

   

