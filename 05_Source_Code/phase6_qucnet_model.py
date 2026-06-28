import tensorflow as tf
from tensorflow.keras import layers, models


def build_qucnet_tumor_model(input_shape=(64, 64, 4), num_classes=2):
    """
    QuCNet tumor classification model.

    Paper pipeline:
    QFT feature maps -> Flatten -> Dense -> Dense -> Softmax
    """

    model = models.Sequential(name="QuCNet_Tumor_Classifier")

    model.add(layers.Input(shape=input_shape))

    # Flatten quantum feature maps
    model.add(layers.Flatten(name="flatten_quantum_features"))

    # Dense layer 1
    model.add(layers.Dense(128, activation="relu", name="dense_1"))

    # Dense layer 2
    model.add(layers.Dense(64, activation="relu", name="dense_2"))

    # Output layer: Benign / Malignant
    model.add(layers.Dense(num_classes, activation="softmax", name="output_softmax"))

    return model


if __name__ == "__main__":

    model = build_qucnet_tumor_model()

    model.summary()