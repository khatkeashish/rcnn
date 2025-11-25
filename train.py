# Import libraries
import numpy as np
import os
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from voc2012 import get_labels, getFilepaths, loadImage, loadAnnotation
from utils import Configs, selectiveSearch, bb_intersection_over_union, getROI
from models import Backbone, Model


def createDataset(image_paths, annotation_paths, image_shape, voc_labels):
    images = []
    labels = []
    for idx in tqdm(range(len(image_paths)), desc="Creating dataset"):
        image = loadImage(image_paths[idx])
        annotation = loadAnnotation(annotation_paths[idx])
        X, y = roiExtractor(image, annotation, image_shape, voc_labels)
        images.extend(X)
        labels.extend(y)
        #if idx==10:
            #break
    return images, labels

def roiExtractor(image, annotation, image_shape, voc_labels):
    X = []  #
    y = []  #
    img_classes = [] # all the objec classes in the image
    img_boxes = [] # Corrosponding boxes to the objects

    # Get objects and their respective bounding boxes
    for object in annotation["objects"]:
        obj_class = list(object.keys())[0]
        if obj_class in voc_labels:
            obj_box = list(object.values())[0]
            img_classes.append(obj_class)
            img_boxes.append(obj_box)
    
    # If no objects found in training labels, exit
    if not img_classes:
        return X, y

    max_background_images = 5 * len(img_classes) # get only 5 background images per object
    num_background_images = 0 # counter for bckground images
    
    # Get selective search proposals for the image
    ss_results = selectiveSearch("fast", image)
    background_images = []
    for ss_result in ss_results:
        x1, y1, w, h = ss_result
        ss_box = [x1, y1, x1+w, y1+h]
        
        #Get iou of the ss_box for each object class (filtered) in the image
        iou_list = []
        for idx in range(len(img_classes)):
            iou = bb_intersection_over_union(ss_box, obj_box)
            roi_class = img_classes[idx]
            iou_list.append(iou)

        # Get max of iou and corrosponding label
        iou_max = max(iou_list)
        if iou_max > 0.7:
            # Get the roi and resize it to the input shape of nn
            roi = getROI(image, ss_box) # crop the image
            roi = cv2.resize(roi, (image_shape[0], image_shape[1])) # resize the image
            roi_class = img_classes[iou_list.index(iou_max)]
            roi_label = voc_labels.get(roi_class)
            X.append(roi)
            y.append(roi_label)
        elif iou_max < 0.2 and num_background_images < max_background_images:
            # Get the roi and resize it to the input shape of nn
            roi = getROI(image, ss_box) # crop the image
            roi = cv2.resize(roi, (image_shape[0], image_shape[1])) # resize the image
            background_images.append(roi)
            num_background_images += 1
    # create labels for background images
    background_labels = [0] * len(background_images) # label for background
    X.extend(background_images)
    y.extend(background_labels)
    assert(len(X)==len(y))
    return X, y





if __name__ == '__main__':
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
    data_dir = "VOC2012_train_val/VOC2012_train_val"
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


    # Get the dataset
    image_paths, annotation_paths = getFilepaths(configs.data_dir)
    
    images, labels = createDataset(image_paths, annotation_paths, configs.image_shape, voc_labels)

    # Convert the images in np arrays and split into train/test dataset
    X = np.array(images)
    y = np.array(labels)
    y = tf.keras.utils.to_categorical(labels, len(voc_labels))
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configs.test_size, random_state=42)

    print("Training images = ", len(X_train))
    print("Test images = ", len(X_test))
    print(X_train.shape)
    print(y_train.shape)


    # Create training ddata generator
    # Training data generator
    trdata = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    traindata = trdata.flow(x=X_train, y=y_train)
    # Tests data generator
    tsdata = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False, vertical_flip=False, rotation_range=0)
    testdata = tsdata.flow(x=X_test, y=y_test)


    backbone = Backbone(
        arch="vgg16",
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
        trainable=False,
    )
    backbone_model = backbone.backboneModel()
    backbone_model.compile(optimizer, loss, metrics)
    backbone_model.summary()

    ## Create model

    # Get the sequential model with trainable head
    model = Model(backbone_model=backbone.backboneModel(), output_classes=configs.num_classes, dropout_rate=configs.dropout_rate)
    # Compile the model
    model.compile(
        optimizer=configs.optimizer,
        loss=configs.loss,
        metrics=configs.metrics,
    )

    # Publish the model summary
    model.summary()

    # Train the model
    # checkpoints for saving the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "person-model.h5",
        verbose=1,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    # Train the model
    model.fit(traindata,
            steps_per_epoch=len(X_train) // configs.batch_size,
            epochs=configs.epochs,
            callbacks=[checkpoint,early],
            validation_data=testdata,
            verbose=1, shuffle=True)

    model.save("person_model")

    print()
    print()
    print("Training finished")
