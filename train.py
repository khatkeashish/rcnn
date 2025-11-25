# Import libraries
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

from voc2012 import get_labels
from utils import Configs
from models import Backbone, Model
from preprocessing import preprocess_dataset
import argparse


# Use preprocessing module to prepare dataset





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', dest='out_dir', required=False, help='Directory to write/read cached preprocessed files (default: ./processed)')
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


    # Preprocess dataset (will cache to preprocessed.pkl in dataset folder)
    # Determine processed dir (allow override)
    if args.out_dir:
        processed_dir = os.path.abspath(args.out_dir)
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
    workers = getattr(args, 'workers', None)
    X, y = preprocess_dataset(configs.data_dir, configs.image_shape, voc_labels, out_path=cache_path, workers=workers)
    # labels -> categorical
    y = tf.keras.utils.to_categorical(y, len(voc_labels))
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
