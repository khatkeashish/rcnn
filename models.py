import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# Get the backbone model i.e. vgg16
class Backbone:
    """Get the VGG network as a backbone."""

    def __init__(
        self,
        arch,
        include_top,
        weights,
        input_shape,
        trainable,
    ):

        self.arch = arch
        self.include_top = include_top
        self.weights = weights
        self.input_tensor = None
        self.input_shape = input_shape
        self.pooling = None
        self.classes = 1000
        self.trainable = trainable

    def vgg16(self):
        return tf.keras.applications.VGG16(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def vgg19(self):
        return tf.keras.applications.VGG19(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def resnet50(self):
        return tf.keras.applications.ResNet50(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def resnet101(self):
        return tf.keras.applications.ResNet101(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def densenet121(self):
        return tf.keras.applications.DenseNet121(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def mobilenetV2(self):
        return tf.keras.applications.MobileNetV2(
            include_top=self.include_top,
            weights=self.weights,
            input_tensor=self.input_tensor,
            input_shape=self.input_shape,
            pooling=self.pooling,
            classes=self.classes,
        )

    def backboneModel(self):
        if self.arch == "vgg16":
            model = self.vgg16()
        elif self.arch == "vgg19":
            model = self.vgg19()
        elif self.arch == "resnet50":
            model = self.resnet50()
        elif self.arch == "resnet101":
            model = self.resnet101()
        elif self.arch == "densenet121":
            model = self.densenet121()
        elif self.arch == "mobilenetV2":
            model = self.mobilenetV2()
        else:
            print("Invalid architecture")
            sys.exit(1)

        model.trainable = self.trainable
        return model


def Model(backbone_model, output_classes, dropout_rate):

    model = Sequential()
    model.add(backbone_model)
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(output_classes, activation="softmax"))
    return model


if __name__ == "__main__":
    arch = "densenet121"
    include_top = True
    weights = "imagenet"
    input_shape = (224, 224, 3)
    trainable = False
    optimizer = "Adam"
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

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

    backbone = Backbone(
        arch,
        include_top,
        weights,
        input_shape,
        trainable,
    )
    backbone_model = backbone.backboneModel()
    backbone_model.compile(optimizer, loss, metrics)
    backbone_model.summary()
