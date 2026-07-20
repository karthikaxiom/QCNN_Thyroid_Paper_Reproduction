from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    Resizing,
)
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (128, 128, 1)
MODEL_INPUT_SIZE = (64, 64)
NUM_CLASSES = 6

LEARNING_RATE = 1e-4

CONV_FILTERS_1 = 16
CONV_FILTERS_2 = 32

DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64


def build_classical_tirads_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Build the controlled classical TI-RADS CNN.

    Raw 128x128 grayscale images are resized internally to 64x64 so the
    classifier has the same spatial dimensions used by the quantum model.
    """

    if len(input_shape) != 3:
        raise ValueError(
            f"Expected a three-dimensional input shape, received {input_shape}"
        )

    if num_classes != 6:
        raise ValueError(
            f"This experiment requires six classes, received {num_classes}"
        )

    if learning_rate <= 0:
        raise ValueError(
            f"Learning rate must be positive, received {learning_rate}"
        )

    inputs = Input(
        shape=input_shape,
        name="raw_ultrasound_input",
    )

    x = Resizing(
        height=MODEL_INPUT_SIZE[0],
        width=MODEL_INPUT_SIZE[1],
        name="resize_to_64x64",
    )(inputs)

    x = Conv2D(
        filters=CONV_FILTERS_1,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_1",
    )(x)

    x = MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling_1",
    )(x)

    x = Conv2D(
        filters=CONV_FILTERS_2,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_2",
    )(x)

    x = MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling_2",
    )(x)

    x = Flatten(
        name="flatten",
    )(x)

    x = Dense(
        units=DENSE_UNITS_1,
        activation="relu",
        name="dense_128",
    )(x)

    x = Dense(
        units=DENSE_UNITS_2,
        activation="relu",
        name="dense_64",
    )(x)

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="tirads_softmax_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Phase10_Classical_TI_RADS_CNN",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_information(model: Model) -> None:
    """Display model and experimental information."""

    print("=" * 76)
    print("PHASE 10 : CLASSICAL TI-RADS ABLATION MODEL")
    print("=" * 76)

    print("\nConfiguration")
    print("-" * 76)
    print(f"Raw image input       : {INPUT_SHAPE}")
    print(f"Internal resize       : {MODEL_INPUT_SIZE}")
    print(f"Number of classes     : {NUM_CLASSES}")
    print(f"Learning rate         : {LEARNING_RATE}")

    print("\nArchitecture")
    print("-" * 76)
    print("128x128x1 grayscale ultrasound")
    print("-> Resize to 64x64x1")
    print("-> Conv2D 16")
    print("-> MaxPooling")
    print("-> Conv2D 32")
    print("-> MaxPooling")
    print("-> Flatten")
    print("-> Dense 128")
    print("-> Dense 64")
    print("-> Six-class Softmax")

    print("\nControlled Comparison")
    print("-" * 76)
    print("Phase 9 input : Quantum feature maps")
    print("Phase 10 input: Original grayscale images")
    print("Purpose       : Measure the value or loss caused by quantum filtering")

    print("\nModel Summary")
    print("-" * 76)
    model.summary()

    print("\nThis script does not start training.")
    print("Model created successfully.")
    print("=" * 76)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)

    classical_model = build_classical_tirads_model()

    print_model_information(classical_model)