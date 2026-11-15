"""Tests for Hugging Face deployment compatibility."""

import json
from pathlib import Path

import gradio
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Confirm the installed scikit-learn version matches the saved model.
def test_scikit_learn_version_matches_model():
    """Check installed sklearn, metadata, and requirements versions."""
    model_info_path = PROJECT_ROOT / "models" / "model_info.json"
    requirements_path = PROJECT_ROOT / "requirements.txt"

    model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()

    expected_version = model_info["scikit_learn_version"]

    assert sklearn.__version__ == expected_version
    assert f"scikit-learn=={expected_version}" in requirements


# Confirm Gradio is installed at the version specified in the README.
def test_hugging_face_configuration():
    """Check Gradio version, Space settings, and application file."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    dev_requirements = (PROJECT_ROOT / "requirements-dev.txt").read_text(
        encoding="utf-8"
    )

    gradio_version = next(
        line.split("==")[1]
        for line in dev_requirements.splitlines()
        if line.startswith("gradio==")
    )

    assert gradio.__version__ == gradio_version
    assert f"sdk_version: {gradio_version}" in readme
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert 'python_version: "3.12"' in readme
    assert (PROJECT_ROOT / "app.py").is_file()


# Confirm the runtime requirements do not install Gradio.
def test_gradio_is_not_in_runtime_requirements():
    """Keep Gradio out of requirements.txt as required for the Space."""
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    package_names = [
        line.split("==")[0].strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "gradio" not in package_names
