from __future__ import annotations

import argparse
import json

import tensorflow as tf

from src.config import load_config
from src.data import load_training_datasets
from src.labels import CLASS_NAMES, save_class_names
from src.model import build_cnn_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plant disease CNN.")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    train_ds, val_ds, class_names = load_training_datasets(config)

    if class_names != CLASS_NAMES:
        print("Warning: dataset class order differs from the fallback CLASS_NAMES list.")

    model = build_cnn_model(
        image_size=config.image_size,
        num_classes=len(class_names),
        learning_rate=config.learning_rate,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    model.save(config.model_path)
    save_class_names(config.class_names_path, class_names)
    with config.history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    print(f"Saved model to {config.model_path}")
    print(f"Saved class names to {config.class_names_path}")
    print(f"Saved history to {config.history_path}")


if __name__ == "__main__":
    main()
