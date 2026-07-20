from __future__ import annotations

import os

# Reduce unnecessary TensorFlow information messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Model configuration

INPUT_SHAPE = (64, 64, 4)
NUM_CLASSES = 6

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.40
DEFAULT_L2_RATE = 1e-4


def convolution_block(
    inputs: tf.Tensor,
    filters: int,
    block_name: str,
) -> tf.Tensor:
    """
    Create one spatial feature-extraction block.

    Structure:
        Conv2D
        Batch Normalization
        ReLU
        Max Pooling
    """

    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(DEFAULT_L2_RATE),
        name=f"{block_name}_conv",
    )(inputs)

    x = BatchNormalization(
        name=f"{block_name}_batch_norm",
    )(x)

    x = Activation(
        "relu",
        name=f"{block_name}_relu",
    )(x)

    x = MaxPooling2D(
        pool_size=(2, 2),
        name=f"{block_name}_max_pool",
    )(x)

    return x


def build_spatial_qucnet(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> Model:
    """
    Build and compile the improved spatial QuCNet classifier.

    The model applies convolution directly to the quantum feature maps,
    instead of flattening them immediately.

    Parameters
    ----------
    input_shape:
        Shape of one quantum feature map.

    num_classes:
        Number of TI-RADS output classes.

    learning_rate:
        Adam optimizer learning rate.

    dropout_rate:
        Dropout rate before the output layer.

    Returns
    -------
    tensorflow.keras.Model
        Compiled spatial QuCNet model.
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

    # Preserve and learn spatial relationships in the quantum feature maps.
    x = convolution_block(
        inputs=inputs,
        filters=16,
        block_name="spatial_block_1",
    )

    x = convolution_block(
        inputs=x,
        filters=32,
        block_name="spatial_block_2",
    )

    x = convolution_block(
        inputs=x,
        filters=64,
        block_name="spatial_block_3",
    )

    # Replaces the very large flattening operation.
    x = GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(DEFAULT_L2_RATE),
        name="dense_64",
    )(x)

    x = Dropout(
        rate=dropout_rate,
        name="classification_dropout",
    )(x)

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="tirads_classification_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Phase9_Spatial_QuCNet",
    )

    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_information(model: Model) -> None:
    """Print model configuration and architecture."""

    print("=" * 72)
    print("PHASE 9 : IMPROVED SPATIAL QUCNET MODEL")
    print("=" * 72)

    print(f"\nInput Shape       : {INPUT_SHAPE}")
    print(f"Number of Classes : {NUM_CLASSES}")
    print(f"Learning Rate     : {DEFAULT_LEARNING_RATE}")
    print(f"Dropout Rate      : {DEFAULT_DROPOUT_RATE}")
    print(f"L2 Rate           : {DEFAULT_L2_RATE}")

    print("\nTI-RADS Class Mapping")
    print("-" * 72)
    print("0 -> TI-RADS 2")
    print("1 -> TI-RADS 3")
    print("2 -> TI-RADS 4a")
    print("3 -> TI-RADS 4b")
    print("4 -> TI-RADS 4c")
    print("5 -> TI-RADS 5")

    print("\nArchitecture")
    print("-" * 72)
    print("Input 64x64x4")
    print("-> Conv2D 16 + BatchNorm + ReLU + MaxPool")
    print("-> Conv2D 32 + BatchNorm + ReLU + MaxPool")
    print("-> Conv2D 64 + BatchNorm + ReLU + MaxPool")
    print("-> Global Average Pooling")
    print("-> Dense 64")
    print("-> Dropout")
    print("-> Six-class Softmax")

    print("\nModel Summary")
    print("-" * 72)
    model.summary()

    print("\nScientific Classification")
    print("-" * 72)
    print("Paper contribution : Quantum-inspired feature transformation")
    print("Phase 8 baseline   : Flatten-and-dense QuCNet classifier")
    print("Phase 9 improvement: Spatial CNN classifier developed in this project")

    print("\nModel created successfully.")
    print("=" * 72)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)

    spatial_model = build_spatial_qucnet()

    print_model_information(spatial_model)