from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential


def build_cnn_model(
    image_size: tuple[int, int],
    num_classes: int,
    learning_rate: float,
) -> tf.keras.Model:
    height, width = image_size

    model = Sequential(
        [
            Conv2D(32, 3, padding="same", activation="relu", input_shape=(height, width, 3)),
            Conv2D(32, 3, activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(64, 3, padding="same", activation="relu"),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(128, 3, padding="same", activation="relu"),
            Conv2D(128, 3, activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(256, 3, padding="same", activation="relu"),
            Conv2D(256, 3, activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(0.25),
            Flatten(),
            Dense(1500, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
