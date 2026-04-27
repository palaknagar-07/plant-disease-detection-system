# Model Card: Plant Disease Detection CNN

## Model Details

- **Architecture:** Custom TensorFlow/Keras CNN
- **Input size:** 128 x 128 RGB
- **Output:** 38 crop health and disease classes
- **Parameters:** 15,054,794
- **Training epochs:** 10
- **Optimizer:** Adam
- **Learning rate:** 0.0001
- **Class mapping:** Stored in `artifacts/class_names.json`

## Intended Use

This model is intended for portfolio demonstration and educational plant disease classification experiments. It can identify disease-like patterns for the classes present in the training dataset.

## Not Intended For

- Final agricultural treatment decisions
- Diagnosis of unknown diseases
- Production use without field validation
- Images outside the supported crop/disease classes

## Evaluation Summary

| Split | Accuracy |
| --- | ---: |
| Training | 98.18% |
| Validation | 95.16% |
| Test | 95.23% |

The test set includes 10,849 images across 38 classes.

## Per-Class Watchlist

The headline accuracy hides a few weaker classes that should be reviewed before claiming field reliability:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Corn Cercospora leaf spot / Gray leaf spot | 0.932 | 0.667 | 0.777 | 102 |
| Potato healthy | 0.913 | 0.700 | 0.792 | 30 |
| Tomato Early blight | 0.821 | 0.805 | 0.813 | 200 |

Macro F1 is 0.933 and weighted F1 is 0.952.

## Known Risks

- The dataset distribution may not represent real field conditions.
- Some classes have much fewer images than others.
- High test accuracy may not transfer to images with complex backgrounds.
- The model has no uncertainty rejection for out-of-distribution inputs.
- The current app performs classification only; it does not localize disease regions or explain predictions.

## Recommended Improvements

- Add Grad-CAM heatmaps for interpretability.
- Collect real mobile-camera validation images.
- Add top-k confidence thresholding.
- Compare against transfer-learning baselines.
- Track experiments with MLflow, Weights & Biases, or TensorBoard.
