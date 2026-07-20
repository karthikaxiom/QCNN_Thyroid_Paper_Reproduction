"""
Phase 9 - Corrected Paper-Structured TI-RADS QuCNet Model

Purpose
-------
Implement the classical classifier structure described for the
six-class suspicious-level classification task in the QuCNet paper.

Paper-described structure:
    Quantum feature input
    Conv2D
    MaxPooling
    Conv2D
    MaxPooling
    Flatten
    Dense
    Dense
    Softmax output

Important reproducibility note
------------------------------
The paper describes the sequence of layers but does not clearly report
the number of convolutional filters, dense neurons, kernel dimensions,
or initial learning rate.

The following are therefore documented implementation decisions:

    Conv2D filters : 16 and 32
    Kernel size    : 3 x 3
    Pooling size   : 2 x 2
    Dense neurons  : 128 and 64
    Learning rate  : 1e-4

This file does not regenerate quantum features and does not train the
model. It only defines and verifies the architecture.
"""

from __future__ import annotations

import os

# Reduce unnecessary TensorFlow logging.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam


# Fixed experiment configuration

INPUT_SHAPE = (64, 64, 4)
NUM_CLASSES = 6

CONV_FILTERS_1 = 16
CONV_FILTERS_2 = 32

KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64

DEFAULT_LEARNING_RATE = 1e-4


def build_paper_tirads_qucnet(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Model:
    """
    Build and compile the corrected paper-structured TI-RADS model.

    Parameters
    ----------
    input_shape:
        Shape of one quantum-transformed image.

    num_classes:
        Number of TI-RADS classes.

    learning_rate:
        Learning rate for the Adam optimizer.

    Returns
    -------
    tensorflow.keras.Model
        Compiled six-class TI-RADS classifier.
    """

    if len(input_shape) != 3:
        raise ValueError(
            "input_shape must contain height, width and channels. "
            f"Received: {input_shape}"
        )

    if num_classes != 6:
        raise ValueError(
            "This experiment is designed for six TI-RADS classes. "
            f"Received: {num_classes}"
        )

    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive. Received: {learning_rate}"
        )

    inputs = Input(
        shape=input_shape,
        name="quantum_feature_input",
    )

    # First Conv2D-MaxPooling pair described in the paper.
    x = Conv2D(
        filters=CONV_FILTERS_1,
        kernel_size=KERNEL_SIZE,
        padding="same",
        activation="relu",
        name="conv2d_1",
    )(inputs)

    x = MaxPooling2D(
        pool_size=POOL_SIZE,
        name="max_pooling_1",
    )(x)

    # Second Conv2D-MaxPooling pair described in the paper.
    x = Conv2D(
        filters=CONV_FILTERS_2,
        kernel_size=KERNEL_SIZE,
        padding="same",
        activation="relu",
        name="conv2d_2",
    )(x)

    x = MaxPooling2D(
        pool_size=POOL_SIZE,
        name="max_pooling_2",
    )(x)

    # Paper describes flattening after the two convolution blocks.
    x = Flatten(
        name="flatten",
    )(x)

    # Two dense layers described for final classification.
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
        name="Phase9_Paper_Structured_TI_RADS_QuCNet",
    )

    optimizer = Adam(
        learning_rate=learning_rate,
    )

    # Sparse categorical cross-entropy is mathematically equivalent to
    # categorical cross-entropy when integer labels are used.
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_information(model: Model) -> None:
    """Print architecture and reproducibility information."""

    print("=" * 76)
    print("PHASE 9 : CORRECTED PAPER-STRUCTURED TI-RADS QUCNET")
    print("=" * 76)

    print("\nInput Configuration")
    print("-" * 76)
    print(f"Input shape       : {INPUT_SHAPE}")
    print(f"Number of classes : {NUM_CLASSES}")
    print(f"Learning rate     : {DEFAULT_LEARNING_RATE}")

    print("\nTI-RADS Class Mapping")
    print("-" * 76)
    print("0 -> TI-RADS 2")
    print("1 -> TI-RADS 3")
    print("2 -> TI-RADS 4a")
    print("3 -> TI-RADS 4b")
    print("4 -> TI-RADS 4c")
    print("5 -> TI-RADS 5")

    print("\nPaper-Described Architecture")
    print("-" * 76)
    print("Quantum feature input: 64x64x4")
    print("-> Conv2D")
    print("-> MaxPooling")
    print("-> Conv2D")
    print("-> MaxPooling")
    print("-> Flatten")
    print("-> Dense")
    print("-> Dense")
    print("-> Six-class Softmax")

    print("\nDocumented Implementation Decisions")
    print("-" * 76)
    print(f"First convolution filters : {CONV_FILTERS_1}")
    print(f"Second convolution filters: {CONV_FILTERS_2}")
    print(f"Convolution kernel size   : {KERNEL_SIZE}")
    print(f"Pooling window size       : {POOL_SIZE}")
    print(f"First dense layer         : {DENSE_UNITS_1}")
    print(f"Second dense layer        : {DENSE_UNITS_2}")

    print("\nModel Summary")
    print("-" * 76)
    model.summary()

    print("\nExperiment Classification")
    print("-" * 76)
    print("Paper contribution:")
    print("  Quantum filter transformation and layer sequence.")

    print("\nImplementation decisions:")
    print("  Filter counts, kernel dimensions, dense neurons,")
    print("  learning rate and TensorFlow implementation.")

    print("\nThis script does not train or regenerate quantum features.")
    print("Model created successfully.")
    print("=" * 76)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)

    paper_model = build_paper_tirads_qucnet()

    print_model_information(paper_model)