"""Starts the dashboard headlessly WITHOUT a trained model, Spark, HDFS or a GPU."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, r"{root}")
    import generate_data
    daily, zones = generate_data.generate_dataset(n_zones=8, n_days=300, seed=1, test_days=60)
    generate_data.save_dataset(daily, zones)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(r"{root}/app.py", default_timeout=600)
    at.run()
    pages = [p for p in at.sidebar.radio(key="page").options]
    bad = []
    for p in pages:
        at.sidebar.radio(key="page").set_value(p).run()
        if len(at.exception) or len(at.error):
            bad.append((p, [e.value[:200] for e in list(at.exception) + list(at.error)]))
    print("BAD:", bad)
    sys.exit(1 if bad else 0)
""")


def test_all_pages_render_without_model(tmp_path):
    env = {"PATH": "", "WASTE_DATA_DIR": str(tmp_path / "d"), "WASTE_MODELS_DIR": str(tmp_path / "m"),
           "WASTE_OUTPUTS_DIR": str(tmp_path / "o")}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run([sys.executable, "-c", SCRIPT.format(root=ROOT)], env=env, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
