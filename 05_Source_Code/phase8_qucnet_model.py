
from __future__ import annotations

import os

# Reduce unnecessary TensorFlow console messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam


# Model configuration

INPUT_SHAPE = (64, 64, 4)
NUM_CLASSES = 6

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT_RATE = 0.30


def build_tirads_qucnet(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> Model:
    """
    Build and compile the six-class TI-RADS QuCNet classifier.

    Parameters
    ----------
    input_shape:
        Shape of one quantum feature map.

    num_classes:
        Number of output classes.

    learning_rate:
        Learning rate used by the Adam optimizer.

    dropout_rate:
        Dropout applied after the dense layers to reduce overfitting.

    Returns
    -------
    tensorflow.keras.Model
        Compiled TI-RADS QuCNet model.
    """

    if len(input_shape) != 3:
        raise ValueError(
            f"input_shape must contain three dimensions, received {input_shape}"
        )

    if num_classes < 2:
        raise ValueError(
            f"num_classes must be at least 2, received {num_classes}"
        )

    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive, received {learning_rate}"
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            f"dropout_rate must be between 0 and 1, received {dropout_rate}"
        )

    inputs = Input(
        shape=input_shape,
        name="quantum_feature_input",
    )

    x = Flatten(name="flatten_quantum_features")(inputs)

    x = Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_128",
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout_128",
    )(x)

    x = Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_64",
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout_64",
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="tirads_classification_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="TI_RADS_QuCNet",
    )

    optimizer = Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model


def print_model_information(model: Model) -> None:
    """Print model and experiment configuration."""

    print("=" * 70)
    print("PHASE 8 : TI-RADS QUCNET MODEL")
    print("=" * 70)

    print(f"\nInput Shape       : {INPUT_SHAPE}")
    print(f"Number of Classes : {NUM_CLASSES}")
    print(f"Learning Rate     : {DEFAULT_LEARNING_RATE}")
    print(f"Dropout Rate      : {DEFAULT_DROPOUT_RATE}")

    print("\nTI-RADS Class Mapping")
    print("-" * 70)
    print("0 -> TI-RADS 2")
    print("1 -> TI-RADS 3")
    print("2 -> TI-RADS 4a")
    print("3 -> TI-RADS 4b")
    print("4 -> TI-RADS 4c")
    print("5 -> TI-RADS 5")

    print("\nModel Summary")
    print("-" * 70)
    model.summary()

    print("\nModel created successfully.")
    print("=" * 70)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)

    tirads_model = build_tirads_qucnet()

    print_model_information(tirads_model)