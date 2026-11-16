"""Regression checks for navigation and batch-result lifetime."""
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_pages_load():
    app = AppTest.from_file(APP, default_timeout=60).run()
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, [e.message for e in app.exception]


def test_batch_results_survive_rerun_and_clear_on_mode_change():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Batch scoring").run()
    app.button[0].click().run()
    app.button[1].click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    expected = app.session_state["batch_results"].copy()
    assert len(expected) == 15
    app.run()
    pd.testing.assert_frame_equal(app.session_state["batch_results"], expected)
    assert len(app.get("download_button")) == 1
    mode = next(r for r in app.radio if r.label == "Prediction Mode")
    mode.set_value(mode.options[1]).run()
    assert len(app.get("download_button")) == 0


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_csv_upload_scores(delimiter):
    sample = pd.read_csv(Path(APP).parent / "data/bank-full.csv", sep=";").head(2)
    upload = BytesIO(sample.drop(columns="y").to_csv(index=False, sep=delimiter).encode())
    with patch("streamlit.file_uploader", return_value=upload):
        app = AppTest.from_file(APP, default_timeout=60).run()
        app.sidebar.radio[0].set_value("Batch scoring").run()
        assert app.button[0].disabled
        app.button[1].click().run()
        assert not app.exception
        assert not app.error, [e.value for e in app.error]
        assert len(app.session_state["batch_results"]) == 2


def test_invalid_upload_does_not_score_previous_samples():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Batch scoring").run()
    app.button[0].click().run()
    with patch("streamlit.file_uploader", return_value=BytesIO(b"\xff")):
        app.run()
        assert not app.exception
        assert app.error
        assert not app.dataframe
        assert len(app.button) == 1


def test_header_only_upload_shows_empty_message():
    with patch("streamlit.file_uploader", return_value=BytesIO(b"age,job\n")):
        app = AppTest.from_file(APP, default_timeout=60).run()
        app.sidebar.radio[0].set_value("Batch scoring").run()
        assert not app.exception
        assert app.warning
        assert len(app.get("download_button")) == 0
