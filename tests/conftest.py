"""Shared pytest fixtures for Voyage Analytics tests.

A recurring source of test flakiness is the singleton model cache in
``src.model.loader.ModelLoader``: tests that load a dummy/real model leave it
cached, which then leaks into subsequent tests (causing ``/predict`` to run
against a model whose input schema does not match).

This module registers an ``autouse`` fixture that clears the singleton model
cache after every test, so each test starts with a cold cache and loads the
artifact referenced by the (per-test) ``settings.model_path``.
"""

import pytest

from src.model.loader import ModelLoader
from src.model.gender_loader import GenderModelLoader
from src.services.recommendation_service import _Catalog


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Ensure the shared model/catalog caches are empty at the start of every test."""
    # Start from a clean slate so a leftover cached model cannot leak in.
    ModelLoader.unload_model()
    GenderModelLoader.unload_model()
    _Catalog.reset()
    yield
    # Always tear down the caches so they cannot leak into the next test.
    ModelLoader.unload_model()
    GenderModelLoader.unload_model()
    _Catalog.reset()
