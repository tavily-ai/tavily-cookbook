"""Pytest configuration for agent-toolkit tests."""

import pytest

pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
