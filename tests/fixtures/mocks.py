"""Shared mock primitives for unit tests.

Centralizing these here keeps test-only dependencies out of individual test
files — every test imports mock objects from `tests.fixtures.mocks` rather
than reaching into Python's `unittest.mock` directly. If we ever swap mocking
libraries (e.g. to `pytest-mock`) the change happens in one place.
"""

from unittest.mock import MagicMock, patch

__all__ = ["MagicMock", "patch"]
