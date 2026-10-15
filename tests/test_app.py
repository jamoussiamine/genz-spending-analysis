"""Tests de fumée : chaque page de l'application s'affiche sans exception ni erreur."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TIMEOUT = 180


def run_page(page=None):
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    if page is not None:
        app.sidebar.radio[0].set_value(page).run()
    return app


def page_names():
    return run_page().sidebar.radio[0].options


def test_home_page_loads():
    app = run_page()
    assert not app.exception
    assert not app.error


@pytest.mark.parametrize("page", page_names())
def test_page_loads(page):
    app = run_page(page)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
