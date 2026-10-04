"""Every job type the worker knows has a task module with a run()."""

from __future__ import annotations

import importlib

from app.cache import make_cache_key
from app.config import MODEL_VERSION
from app.main import registration_key
from app.worker import TASKS


def test_every_model_has_a_runnable_task():
    for model, module in TASKS.items():
        task = importlib.import_module(f"app.tasks.{module}")
        assert callable(task.run), model


def test_lesion_change_finds_the_register_jobs_transform():
    # the transform is saved under the register job's cache key
    assert registration_key("cur", "pri") == make_cache_key(
        "cur", "register", MODEL_VERSION, {"prior_series_uid": "pri"}
    )
