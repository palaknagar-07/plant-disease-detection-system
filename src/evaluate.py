from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config
from src.data import load_test_dataset
from src.labels import load_class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained plant disease model.")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    config = load_config(args.config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(config.model_path)
    test_ds, dataset_class_names = load_test_dataset(config)
    class_names = load_class_names(config.class_names_path)

    if class_names != dataset_class_names:
        raise ValueError(
            "Class names artifact does not match the test dataset folder order. "
            "Retrain the model or regenerate artifacts/class_names.json."
        )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
    }

    metrics_path = config.reports_dir / "metrics.json"
    cm_path = config.reports_dir / "confusion_matrix.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    np.savetxt(cm_path, matrix, delimiter=",", fmt="%d")

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
