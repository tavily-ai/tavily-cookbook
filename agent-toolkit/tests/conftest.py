"""Pytest configuration for agent-toolkit tests."""

import sys
from pathlib import Path

import pytest

# Add agent-toolkit/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
