"""Waste image classification interface (Modules 1-3).

THIS PROJECT DOES NOT TRAIN OR SHIP A MODEL. You train a MobileNetV2 classifier yourself
and place two files in ``models/``:

    models/waste_classifier.keras     your trained Keras model
    models/class_names.json           the six class names, in the model's output order

A prediction is shown ONLY when a valid trained model is loaded, its output passes
validation, and the confidence reaches the threshold (default 0.80). Otherwise a clear
status is returned and NO category is presented as an AI result.

TensorFlow is imported lazily, so the rest of the app runs without it.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

import config


class Status(str, Enum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_MISSING = "model_missing"
    CLASS_NAMES_MISSING = "class_names_missing"
    CLASS_NAMES_INVALID = "class_names_invalid"
    TENSORFLOW_MISSING = "tensorflow_missing"
    MODEL_LOAD_ERROR = "model_load_error"
    INVALID_IMAGE = "invalid_image"
    INVALID_OUTPUT = "invalid_output"
    ERROR = "error"


@dataclass
class ClassificationResult:
    status: Status
    message: str
    label: str | None = None             # set ONLY when status == OK
    confidence: float | None = None      # set ONLY when status in (OK, LOW_CONFIDENCE)
    probabilities: dict[str, float] = field(default_factory=dict)  # real model output only
    top_label_unaccepted: str | None = None   # set for LOW_CONFIDENCE (not an accepted label)
    latency_ms: float | None = None

    @property
    def accepted(self) -> bool:
        return self.status == Status.OK


SETUP_INSTRUCTIONS = (
    "To enable image classification:\n"
    f"1. Install the optional ML dependencies:  pip install -r requirements-ml.txt\n"
    f"2. Train your own MobileNetV2 model ({config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} RGB, 6 classes, softmax output).\n"
    f"3. Save it as  models/{config.MODEL_PATH.name}\n"
    f"4. Save the class names (output order) as  models/{config.CLASS_NAMES_PATH.name},  e.g.\n"
    '   ["Paper", "Plastic", "Glass", "Metal", "Organic", "Other"]\n'
    "5. Restart the dashboard."
)


# ---------------------------------------------------------------------------
# Image validation and preprocessing (no TensorFlow needed)
# ---------------------------------------------------------------------------
class ImageValidationError(ValueError):
    pass


def validate_image(data: bytes):
    """Validate raw upload bytes and return an RGB PIL image. Raises ImageValidationError."""
    from PIL import Image, ImageOps  # Pillow is a core dependency

    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageValidationError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()                       # detects truncated / corrupt files
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ImageValidationError(f"This file is not a readable image ({type(exc).__name__}).") from exc
    if fmt not in config.ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError(f"Unsupported image format '{fmt}'. Allowed: {', '.join(config.ALLOWED_IMAGE_FORMATS)}.")
    if min(img.size) < config.MIN_IMAGE_SIDE_PX:
        raise ImageValidationError(f"Image is too small ({img.size[0]}x{img.size[1]} px); "
                                   f"minimum side is {config.MIN_IMAGE_SIDE_PX} px.")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")


def preprocess_image(img, mode: str | None = None, denoise: bool | None = None) -> np.ndarray:
    """Resize to 224x224 RGB and scale. Returns float32 array of shape (1, 224, 224, 3).

    The scaling MUST match how you trained your model (config.PREPROCESS_MODE).
    """
    from PIL import Image, ImageFilter

    mode = mode or config.PREPROCESS_MODE
    denoise = config.APPLY_DENOISE if denoise is None else denoise
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(3))
    img = img.resize(config.IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape != (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3):
        raise ImageValidationError("Unexpected image shape after preprocessing.")
    if mode == "mobilenet_v2":
        arr = arr / 127.5 - 1.0
    elif mode == "rescale_0_1":
        arr = arr / 255.0
    elif mode != "none":
        raise ValueError(f"Unknown PREPROCESS_MODE '{mode}'")
    return arr[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Class names and model files
# ---------------------------------------------------------------------------
def _canonical(name: str) -> str | None:
    for c in config.CATEGORIES:
        if str(name).strip().lower() == c.lower():
            return c
    return None


def load_class_names(path: Path | None = None) -> list[str]:
    """Load and validate class_names.json. Raises FileNotFoundError / ValueError."""
    path = Path(path) if path else config.CLASS_NAMES_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    if isinstance(raw, dict):                           # {"0": "Paper", ...}
        try:
            raw = [raw[k] for k in sorted(raw, key=lambda x: int(x))]
        except Exception as exc:
            raise ValueError("class_names.json dict keys must be integer indices.") from exc
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("class_names.json must be a list of strings (or {index: name}).")
    if len(raw) != len(config.CATEGORIES):
        raise ValueError(f"Expected {len(config.CATEGORIES)} class names, found {len(raw)}.")
    canon = [_canonical(x) for x in raw]
    if any(c is None for c in canon) or len(set(canon)) != len(canon):
        raise ValueError("Class names must be exactly these six (any case, each once): "
                         + ", ".join(config.CATEGORIES))
    return canon  # type: ignore[return-value]


def check_files() -> dict:
    """Cheap status check (does not import TensorFlow)."""
    import importlib.util
    info = {"model_path": str(config.MODEL_PATH), "class_names_path": str(config.CLASS_NAMES_PATH),
            "model_exists": config.MODEL_PATH.exists(), "class_names_exists": config.CLASS_NAMES_PATH.exists(),
            "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
            "class_names_valid": None, "class_names_error": None}
    if info["class_names_exists"]:
        try:
            load_class_names()
            info["class_names_valid"] = True
        except Exception as exc:
            info["class_names_valid"] = False
            info["class_names_error"] = str(exc)
    return info


# ---------------------------------------------------------------------------
# Classifier service
# ---------------------------------------------------------------------------
class ClassifierService:
    """Loads the user's trained model lazily and classifies validated images."""

    def __init__(self, model_path: Path | None = None, class_names_path: Path | None = None,
                 threshold: float = config.CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        self.class_names_path = Path(class_names_path) if class_names_path else config.CLASS_NAMES_PATH
        self.threshold = threshold
        self._model = None
        self.class_names: list[str] | None = None
        self.load_status: Status | None = None
        self.load_message: str = ""
        self.tf_version: str | None = None

    # -- loading -----------------------------------------------------------
    def load(self) -> bool:
        """Try to load model + class names. Returns True when ready to predict."""
        if self._model is not None and self.class_names:
            return True
        if not self.model_path.exists():
            return self._fail(Status.MODEL_MISSING, f"Model file not found: {self.model_path.name}.")
        try:
            self.class_names = load_class_names(self.class_names_path)
        except FileNotFoundError as exc:
            return self._fail(Status.CLASS_NAMES_MISSING, str(exc))
        except ValueError as exc:
            return self._fail(Status.CLASS_NAMES_INVALID, str(exc))
        try:
            import tensorflow as tf  # lazy, optional dependency
        except Exception as exc:  # ImportError or DLL problems on Windows
            return self._fail(Status.TENSORFLOW_MISSING,
                              f"TensorFlow could not be imported ({type(exc).__name__}). "
                              "Install it with: pip install -r requirements-ml.txt")
        try:
            self._model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.tf_version = getattr(tf, "__version__", None)
        except Exception as exc:
            return self._fail(Status.MODEL_LOAD_ERROR, f"Could not load the model: {exc}")
        n_out = self._output_units()
        if n_out is not None and n_out != len(self.class_names):
            self._model = None
            return self._fail(Status.CLASS_NAMES_INVALID,
                              f"Model has {n_out} outputs but class_names.json lists {len(self.class_names)} classes.")
        self.load_status, self.load_message = Status.OK, "Model loaded."
        return True

    def _fail(self, status: Status, message: str) -> bool:
        self.load_status, self.load_message = status, message
        return False

    def _output_units(self) -> int | None:
        try:
            shape = self._model.output_shape
            return int(shape[-1]) if shape and shape[-1] else None
        except Exception:
            return None

    # -- prediction --------------------------------------------------------
    def classify_bytes(self, data: bytes) -> ClassificationResult:
        try:
            img = validate_image(data)
            batch = preprocess_image(img)
        except ImageValidationError as exc:
            return ClassificationResult(Status.INVALID_IMAGE, str(exc))
        except Exception as exc:
            return ClassificationResult(Status.INVALID_IMAGE, f"Image could not be processed: {exc}")
        if not self.load():
            return ClassificationResult(self.load_status or Status.ERROR, self.load_message)
        return self.predict_batch(batch)

    def predict_batch(self, batch: np.ndarray) -> ClassificationResult:
        """Run the loaded model on a preprocessed batch and validate its output."""
        if self._model is None or not self.class_names:
            return ClassificationResult(self.load_status or Status.MODEL_MISSING, self.load_message or "No model loaded.")
        try:
            t0 = time.perf_counter()
            out = np.asarray(self._model.predict(batch, verbose=0), dtype=np.float64)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            return ClassificationResult(Status.ERROR, f"Model inference failed: {exc}")
        return self.interpret_output(out, latency)

    def interpret_output(self, out: np.ndarray, latency_ms: float | None = None) -> ClassificationResult:
        """Validate raw model output and apply the confidence threshold."""
        n = len(self.class_names or [])
        if out.shape != (1, n):
            return ClassificationResult(Status.INVALID_OUTPUT, f"Unexpected model output shape {out.shape}; expected (1, {n}).")
        if not np.all(np.isfinite(out)):
            return ClassificationResult(Status.INVALID_OUTPUT, "Model output contains NaN or infinite values.")
        probs = out[0]
        tol = config.OUTPUT_SUM_TOLERANCE
        if probs.min() < -tol or probs.max() > 1 + tol or abs(probs.sum() - 1.0) > tol:
            return ClassificationResult(
                Status.INVALID_OUTPUT,
                "Model output is not a probability distribution (values must lie in [0, 1] and sum to 1). "
                "Make sure the final layer uses softmax.")
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.class_names[idx]
        dist = {name: float(p) for name, p in zip(self.class_names, probs)}
        if conf < self.threshold:
            return ClassificationResult(
                Status.LOW_CONFIDENCE,
                f"Confidence {conf:.1%} is below the {self.threshold:.0%} threshold - route to manual review.",
                None, conf, dist, label, latency_ms)
        return ClassificationResult(Status.OK, "Prediction accepted.", label, conf, dist, None, latency_ms)
