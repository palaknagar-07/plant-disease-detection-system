from __future__ import annotations

import json
from pathlib import Path


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_class_names(path: str | Path) -> list[str]:
    class_names_path = Path(path)
    if not class_names_path.exists():
        return CLASS_NAMES

    with class_names_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError(f"Invalid class names artifact: {class_names_path}")

    return class_names


def save_class_names(path: str | Path, class_names: list[str]) -> None:
    class_names_path = Path(path)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)

    with class_names_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def humanize_label(label: str) -> str:
    crop, _, condition = label.partition("___")
    crop = crop.replace("_", " ")
    condition = condition.replace("_", " ")
    return f"{crop}: {condition}"


def is_healthy(label: str) -> bool:
    return label.endswith("___healthy")
