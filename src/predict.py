from __future__ import annotations

import argparse
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.labels import load_class_names, humanize_label


def load_image_array(image_path: str | Path | BinaryIO, image_size: tuple[int, int]) -> np.ndarray:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def predict_top_k(
    model: tf.keras.Model,
    image_path: str | Path | BinaryIO,
    image_size: tuple[int, int],
    class_names: list[str],
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    image_array = load_image_array(image_path, image_size)
    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = probabilities.argsort()[-top_k:][::-1]

    return [
        {
            "class_name": class_names[index],
            "display_name": humanize_label(class_names[index]),
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict plant disease for one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    model = tf.keras.models.load_model(config.model_path)
    class_names = load_class_names(config.class_names_path)
    predictions = predict_top_k(model, args.image_path, config.image_size, class_names, config.top_k)

    for rank, prediction in enumerate(predictions, start=1):
        confidence = prediction["confidence"] * 100
        print(f"{rank}. {prediction['display_name']} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()
