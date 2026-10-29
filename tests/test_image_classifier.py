"""Image-classifier tests.

NOTE: the stub model below exists ONLY inside these unit tests to exercise the validation
and threshold logic. It is not a trained model and the app never uses stubs.
"""
import io
import json

import numpy as np
import pytest
from PIL import Image

import config
import image_classifier as ic


def _png(size=(100, 80), color=(120, 60, 30), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    return buf.getvalue()


class StubModel:
    def __init__(self, out):
        self.out = np.asarray(out, dtype=float)

    def predict(self, batch, verbose=0):
        return self.out


def _service(tmp_path, out=None):
    names = tmp_path / "class_names.json"
    names.write_text(json.dumps(config.CATEGORIES))
    svc = ic.ClassifierService(tmp_path / "m.keras", names)
    svc.class_names = list(config.CATEGORIES)
    if out is not None:
        svc._model = StubModel(out)
    return svc


def test_preprocess_shape_and_range():
    img = ic.validate_image(_png())
    x = ic.preprocess_image(img, mode="mobilenet_v2")
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert ic.preprocess_image(img, mode="rescale_0_1").max() <= 1.0
    assert ic.preprocess_image(img, mode="none").max() > 1.0


def test_rgba_and_large_inputs():
    img = ic.validate_image(_png(mode="RGBA", color=(1, 2, 3, 128)))
    assert img.mode == "RGB"


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"not an image", "not a readable image"),
                                      (_png((10, 10)), "too small")])
def test_invalid_uploads(data, msg):
    with pytest.raises(ic.ImageValidationError, match=msg):
        ic.validate_image(data)


def test_oversized_upload_rejected():
    with pytest.raises(ic.ImageValidationError, match="larger"):
        ic.validate_image(b"0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1))


def test_missing_model_returns_status_not_prediction(tmp_path):
    svc = ic.ClassifierService(tmp_path / "none.keras", tmp_path / "none.json")
    res = svc.classify_bytes(_png())
    assert res.status == ic.Status.MODEL_MISSING
    assert res.label is None and res.confidence is None and not res.probabilities


def test_invalid_image_status(tmp_path):
    res = ic.ClassifierService(tmp_path / "x", tmp_path / "y").classify_bytes(b"junk")
    assert res.status == ic.Status.INVALID_IMAGE


def test_class_names_loading(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["paper", "PLASTIC", "Glass", "Metal", "Organic", "Other"]))
    assert ic.load_class_names(p) == config.CATEGORIES
    p.write_text(json.dumps({str(i): n for i, n in enumerate(config.CATEGORIES)}))
    assert ic.load_class_names(p) == config.CATEGORIES
    p.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError):
        ic.load_class_names(p)
    p.write_text("{not json")
    with pytest.raises(ValueError):
        ic.load_class_names(p)
    with pytest.raises(FileNotFoundError):
        ic.load_class_names(tmp_path / "missing.json")


def test_threshold_logic(tmp_path):
    ok = _service(tmp_path, [[0.05, 0.85, 0.02, 0.03, 0.03, 0.02]]).predict_batch(np.zeros((1, 224, 224, 3)))
    assert ok.status == ic.Status.OK and ok.label == "Plastic" and ok.confidence == pytest.approx(0.85)
    low = _service(tmp_path, [[0.30, 0.50, 0.05, 0.05, 0.05, 0.05]]).predict_batch(np.zeros((1, 224, 224, 3)))
    assert low.status == ic.Status.LOW_CONFIDENCE and low.label is None
    assert low.top_label_unaccepted == "Plastic"


@pytest.mark.parametrize("out", [[[0.5, 0.5]], [[np.nan] * 6], [[2.0, -1.0, 0, 0, 0, 0]],
                                 [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]], [[0.1] * 6, [0.1] * 6]])
def test_invalid_model_output(tmp_path, out):
    res = _service(tmp_path, out).predict_batch(np.zeros((1, 224, 224, 3)))
    assert res.status == ic.Status.INVALID_OUTPUT
    assert res.label is None and res.confidence is None


def test_check_files_does_not_require_tensorflow():
    info = ic.check_files()
    assert "tensorflow_installed" in info and "model_exists" in info
