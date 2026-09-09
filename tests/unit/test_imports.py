"""Importability checks for the ``diary`` package (Task 01)."""

from diary import __version__
from diary.main import app


def test_package_version_is_defined() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_app_is_importable_and_named() -> None:
    assert app is not None
    assert app.title == "Local Diary"
