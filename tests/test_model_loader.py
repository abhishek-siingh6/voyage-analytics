"""Test model loader."""

import pytest
import os
import tempfile

from src.model.loader import ModelLoader, load_model, get_model
from tests.fixtures.dummy_model import DummyPipeline


@pytest.fixture
def dummy_model_path():
    """Create a temporary dummy model for testing."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name

    pipeline = DummyPipeline()
    pipeline.save(path)

    yield path

    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


def test_load_model_succeeds(dummy_model_path):
    """Test model loads successfully from valid path."""
    # Temporarily override model path
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        model = load_model()
        assert model is not None
        assert isinstance(model, DummyPipeline)
    finally:
        config.settings.settings.model_path = original_path


def test_load_model_cache(dummy_model_path):
    """Test model is cached after first load."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        model1 = load_model()
        model2 = load_model()

        # Should return same instance
        assert model1 is model2
    finally:
        config.settings.settings.model_path = original_path


def test_load_model_file_not_found():
    """Test load_model raises FileNotFoundError for missing model."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = "/nonexistent/path/model.joblib"

    # Ensure a previously cached model cannot mask the missing file.
    ModelLoader.unload_model()

    try:
        with pytest.raises(FileNotFoundError):
            load_model()
    finally:
        config.settings.settings.model_path = original_path


def test_get_model_loads_if_not_loaded(dummy_model_path):
    """Test get_model loads model if not already loaded."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        # Clear cache
        ModelLoader.unload_model()

        model = get_model()
        assert model is not None
        assert isinstance(model, DummyPipeline)
    finally:
        config.settings.settings.model_path = original_path


def test_get_model_returns_cached(dummy_model_path):
    """Test get_model returns cached model."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        # First load
        model1 = load_model()

        # get_model should return same instance
        model2 = get_model()

        assert model1 is model2
    finally:
        config.settings.settings.model_path = original_path


def test_unload_model_clears_cache(dummy_model_path):
    """Test unload_model clears the cache."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        load_model()
        ModelLoader.unload_model()

        # Cache should be empty
        assert ModelLoader._model_instance is None
    finally:
        config.settings.settings.model_path = original_path


def test_model_has_expected_attributes(dummy_model_path):
    """Test loaded model has expected attributes."""
    import config.settings

    original_path = config.settings.settings.model_path
    config.settings.settings.model_path = dummy_model_path

    try:
        model = load_model()

        assert hasattr(model, "predict")
        assert hasattr(model, "model_name")
        assert hasattr(model, "model_version")
    finally:
        config.settings.settings.model_path = original_path
