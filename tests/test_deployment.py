"""Tests for Hugging Face deployment compatibility."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scikit_learn_version_matches_model():
    """Check that the model and requirements use the same sklearn version."""
    model_info_path = PROJECT_ROOT / "models" / "model_info.json"
    requirements_path = PROJECT_ROOT / "requirements.txt"

    model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    requirements = requirements_path.read_text(encoding="utf-8")

    expected_version = model_info["scikit_learn_version"]

    assert f"scikit-learn=={expected_version}" in requirements.splitlines()


def test_hugging_face_configuration():
    """Check Gradio version and Hugging Face Space settings."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    dev_requirements = (PROJECT_ROOT / "requirements-dev.txt").read_text(
        encoding="utf-8"
    )

    gradio_version = next(
        line.split("==")[1]
        for line in dev_requirements.splitlines()
        if line.startswith("gradio==")
    )

    assert f"sdk_version: {gradio_version}" in readme
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert 'python_version: "3.12"' in readme
    assert (PROJECT_ROOT / "app.py").is_file()
