"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def simple_hp_sequence():
    """Fixture providing a simple HP sequence."""
    return "HPHPPH"


@pytest.fixture
def medium_hp_sequence():
    """Fixture providing a medium HP sequence."""
    return "HPHPPHHPPHPHHH"
