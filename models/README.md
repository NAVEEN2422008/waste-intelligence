# Where to put YOUR trained model

This project does **not** include a trained model and never trains one.

Place exactly these two files in this folder:

| File | What it is |
|------|-----------|
| `waste_classifier.keras` | Your trained Keras model (MobileNetV2 backbone, 224x224x3 input, 6-way **softmax** output) |
| `class_names.json` | The six class names **in the same order as your model's output units** |

Example `class_names.json`:

```json
["Paper", "Plastic", "Glass", "Metal", "Organic", "Other"]
```

(`{"0": "Paper", "1": "Plastic", ...}` is also accepted. Names must be exactly these six, in any letter case.)

Preprocessing must match training. Default is MobileNetV2 scaling (`x / 127.5 - 1`). If you trained
with a different scaling, or your saved model already contains its own preprocessing layer, change
`PREPROCESS_MODE` in `config.py` (`"mobilenet_v2"`, `"rescale_0_1"` or `"none"`).

Then install TensorFlow (`pip install -r requirements-ml.txt`) and restart the dashboard.
The 'Classify Image' page shows a prediction only when the model loads, its output is a valid
probability vector, and the confidence is at least 0.80.
