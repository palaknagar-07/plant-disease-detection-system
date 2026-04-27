from __future__ import annotations

import tensorflow as tf

from src.config import ProjectConfig


def load_training_datasets(config: ProjectConfig):
    common_args = {
        "labels": "inferred",
        "label_mode": "categorical",
        "color_mode": "rgb",
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "shuffle": True,
        "seed": config.seed,
        "validation_split": config.validation_split,
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.train_dir,
        subset="training",
        **common_args,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.train_dir,
        subset="validation",
        **common_args,
    )
    return _prefetch(train_ds), _prefetch(val_ds), train_ds.class_names


def load_test_dataset(config: ProjectConfig):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
    )
    return _prefetch(test_ds), test_ds.class_names


def _prefetch(dataset):
    return dataset.prefetch(tf.data.AUTOTUNE)
