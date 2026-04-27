# Deployment Notes

This project keeps the dataset and trained model out of Git because both are large generated artifacts. The Streamlit app can still be deployed cleanly by providing the trained model as an external download.

## Recommended Runtime

- Python: 3.11
- App command: `streamlit run main.py`
- Install command: `python3 -m pip install -r requirements.txt`

`runtime.txt` is included for hosts that support Python runtime selection.

## Model Artifact

The app first looks for `trained_model.keras` in the project root. If the file is missing, it checks the `PLANT_MODEL_URL` environment variable and downloads the model from that URL.

Good places to host the model:

- GitHub Releases
- Hugging Face Hub
- Google Drive with a direct-download link

Keep `artifacts/class_names.json` committed with the app. It preserves the exact class order used during training, which prevents predictions from being mapped to the wrong disease labels.

## Common Deployment Failures

- Missing `trained_model.keras` and no `PLANT_MODEL_URL` set.
- Using a Python version that TensorFlow does not support.
- Installing `opencv-python` instead of `opencv-python-headless` on a hosted Linux environment.
- Forgetting to include `artifacts/class_names.json`.
