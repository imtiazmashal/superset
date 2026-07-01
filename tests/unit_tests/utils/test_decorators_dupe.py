# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging
import time
import uuid
from contextlib import nullcontext
from inspect import isclass
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.exc import SQLAlchemyError

from superset.utils import decoraters2

# ---------------------------------------------------------------------------
# statsd_gauge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, response, expected_exception, expected_metric",
    [
        ("pfx", "ok", None, "pfx.ok"),
        ("pfx", "error", ValueError, "pfx.error"),
        ("pfx", "warn", FileNotFoundError, "pfx.warning"),
        (None, "ok", None, "my_func.ok"),
    ],
)
def test_statsd_gauge(
    prefix: Optional[str],
    response: str,
    expected_exception: Optional[type[Exception]],
    expected_metric: str,
) -> None:
    @decoraters2.statsd_gauge(prefix)
    def my_func(resp: str) -> str:
        if resp == "error":
            raise ValueError("boom")
        if resp == "warn":
            exc = FileNotFoundError("missing")
            exc.status = 404  # type: ignore[attr-defined]
            raise exc
        return "OK"

    with patch(
        "superset.utils.decoraters2.app.config",
        {"STATS_LOGGER": Mock()},
    ) as cfg:
        cm = (
            pytest.raises(expected_exception)
            if isclass(expected_exception) and issubclass(expected_exception, Exception)
            else nullcontext()
        )
        with cm:
            my_func(response)

        cfg["STATS_LOGGER"].gauge.assert_called_once_with(expected_metric, 1)


def test_statsd_gauge_error_without_status_attr() -> None:
    @decoraters2.statsd_gauge("pfx")
    def boom() -> None:
        raise RuntimeError("no status attr")

    with patch(
        "superset.utils.decoraters2.app.config",
        {"STATS_LOGGER": Mock()},
    ) as cfg:
        with pytest.raises(RuntimeError):
            boom()
        cfg["STATS_LOGGER"].gauge.assert_called_once_with("pfx.error", 1)


def test_statsd_gauge_error_status_gte_500() -> None:
    @decoraters2.statsd_gauge("pfx")
    def boom() -> None:
        exc = Exception("server error")
        exc.status = 500  # type: ignore[attr-defined]
        raise exc

    with patch(
        "superset.utils.decoraters2.app.config",
        {"STATS_LOGGER": Mock()},
    ) as cfg:
        with pytest.raises(Exception, match="server error"):
            boom()
        cfg["STATS_LOGGER"].gauge.assert_called_once_with("pfx.error", 1)


# ---------------------------------------------------------------------------
# logs_context
# ---------------------------------------------------------------------------


@patch("superset.utils.decoraters2.g")
def test_logs_context_no_kwargs(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context()
    def func(*args: Any, **kwargs: Any) -> str:
        return "ok"

    func(1, 2)
    assert flask_g.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_with_allowed_kwargs(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context()
    def func(*args: Any, **kwargs: Any) -> str:
        return "ok"

    func(slice_id=10, dashboard_id=20)
    assert flask_g.logs_context == {"slice_id": 10, "dashboard_id": 20}


@patch("superset.utils.decoraters2.g")
def test_logs_context_filters_disallowed_keys(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context()
    def func(**kwargs: Any) -> str:
        return "ok"

    func(slice_id=1, random_key="ignored")
    assert flask_g.logs_context == {"slice_id": 1}


@patch("superset.utils.decoraters2.g")
def test_logs_context_filters_none_values(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context()
    def func(**kwargs: Any) -> str:
        return "ok"

    func(slice_id=None, dashboard_id=5)
    assert flask_g.logs_context == {"dashboard_id": 5}


@patch("superset.utils.decoraters2.g")
def test_logs_context_decorator_kwargs_override(flask_g: Mock) -> None:
    uid = uuid.uuid4()
    flask_g.logs_context = {}

    @decoraters2.logs_context(slice_id=99, execution_id=uid)
    def func(**kwargs: Any) -> str:
        return "ok"

    func(slice_id=1)
    assert flask_g.logs_context["slice_id"] == 99
    assert flask_g.logs_context["execution_id"] == uid


@patch("superset.utils.decoraters2.g")
def test_logs_context_with_context_func(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context(
        context_func=lambda *a, **kw: {"slice_id": kw.get("chart_id")}
    )
    def func(**kwargs: Any) -> str:
        return "ok"

    func(chart_id=42)
    assert flask_g.logs_context == {"slice_id": 42}


@patch("superset.utils.decoraters2.g")
def test_logs_context_context_func_overrides_kwargs_and_decorator(
    flask_g: Mock,
) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context(
        slice_id=1,
        context_func=lambda *a, **kw: {"slice_id": 999, "extra_key": "nope"},
    )
    def func(**kwargs: Any) -> str:
        return "ok"

    func(slice_id=5)
    assert flask_g.logs_context == {"slice_id": 999}


@patch("superset.utils.decoraters2.g")
def test_logs_context_bad_context_func_return(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context(context_func=lambda: "not a dict")  # type: ignore[arg-type,return-value]
    def func() -> str:
        return "ok"

    func()
    assert flask_g.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_non_callable_context_func(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context(context_func="not callable")  # type: ignore[arg-type]
    def func() -> str:
        return "ok"

    func()
    assert flask_g.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_initializes_logs_context_on_g(flask_g: Mock) -> None:
    del flask_g.logs_context

    @decoraters2.logs_context()
    def func(**kwargs: Any) -> str:
        return "ok"

    func(slice_id=7)
    assert flask_g.logs_context == {"slice_id": 7}


@patch("superset.utils.decoraters2.g")
def test_logs_context_all_allowed_keys(flask_g: Mock) -> None:
    flask_g.logs_context = {}

    @decoraters2.logs_context()
    def func(**kwargs: Any) -> str:
        return "ok"

    func(
        slice_id=1,
        dashboard_id=2,
        dataset_id=3,
        execution_id=4,
        report_schedule_id=5,
    )
    assert flask_g.logs_context == {
        "slice_id": 1,
        "dashboard_id": 2,
        "dataset_id": 3,
        "execution_id": 4,
        "report_schedule_id": 5,
    }


# ---------------------------------------------------------------------------
# stats_timing
# ---------------------------------------------------------------------------


def test_stats_timing_calls_timing() -> None:
    mock_logger = Mock()
    with decoraters2.stats_timing("my.key", mock_logger) as start:
        assert isinstance(start, float)
    mock_logger.timing.assert_called_once()
    call_args = mock_logger.timing.call_args
    assert call_args[0][0] == "my.key"
    assert call_args[0][1] >= 0


def test_stats_timing_calls_timing_on_exception() -> None:
    mock_logger = Mock()
    with pytest.raises(ValueError, match="boom"):
        with decoraters2.stats_timing("err.key", mock_logger):
            raise ValueError("boom")
    mock_logger.timing.assert_called_once()
    assert mock_logger.timing.call_args[0][0] == "err.key"


# ---------------------------------------------------------------------------
# arghash
# ---------------------------------------------------------------------------


def test_arghash_same_args_same_hash() -> None:
    h1 = decoraters2.arghash((1, "a"), {"k": "v"})
    h2 = decoraters2.arghash((1, "a"), {"k": "v"})
    assert h1 == h2


def test_arghash_different_args_different_hash() -> None:
    h1 = decoraters2.arghash((1,), {"k": "v"})
    h2 = decoraters2.arghash((2,), {"k": "v"})
    assert h1 != h2


def test_arghash_kwargs_order_irrelevant() -> None:
    h1 = decoraters2.arghash((), {"a": 1, "b": 2})
    h2 = decoraters2.arghash((), {"b": 2, "a": 1})
    assert h1 == h2


def test_arghash_empty() -> None:
    h = decoraters2.arghash((), {})
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# debounce
# ---------------------------------------------------------------------------


def test_debounce_suppresses_repeated_calls() -> None:
    inner = Mock()

    @decoraters2.debounce()
    def func(x: int) -> int:
        inner(x)
        return x * 2

    result1 = func(5)
    result2 = func(5)
    assert result1 == 10
    assert result2 == 10
    inner.assert_called_once_with(5)


def test_debounce_calls_again_after_duration() -> None:
    inner = Mock(return_value=1)

    @decoraters2.debounce(duration=0.0)
    def func() -> int:
        return inner()

    func()
    time.sleep(0.01)
    func()
    assert inner.call_count == 2


def test_debounce_calls_again_with_different_args() -> None:
    inner = Mock(side_effect=lambda x: x)

    @decoraters2.debounce()
    def func(x: int) -> int:
        return inner(x)

    assert func(1) == 1
    assert func(2) == 2
    assert inner.call_count == 2


def test_debounce_kwarg_order_does_not_matter() -> None:
    inner = Mock(return_value=0)

    @decoraters2.debounce()
    def func(a: int = 0, b: int = 0) -> int:
        return inner(a, b)

    func(a=1, b=2)
    func(b=2, a=1)
    inner.assert_called_once()


# ---------------------------------------------------------------------------
# on_security_exception
# ---------------------------------------------------------------------------


def test_on_security_exception() -> None:
    mock_self = Mock()
    mock_self.response.return_value = "forbidden"
    ex = PermissionError("nope")

    result = decoraters2.on_security_exception(mock_self, ex)

    assert result == "forbidden"
    mock_self.response.assert_called_once()
    call_args = mock_self.response.call_args
    assert call_args[0][0] == 403
    assert "message" in call_args[1]


# ---------------------------------------------------------------------------
# suppress_logging
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_suppress_logging_default_level() -> None:
    handler = _ListHandler()
    test_logger = logging.getLogger("test_decoraters2_suppress")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    with decoraters2.suppress_logging("test_decoraters2_suppress"):
        test_logger.error("should be hidden")
        test_logger.critical("should show")

    assert len(handler.records) == 1
    assert handler.records[0].levelname == "CRITICAL"
    test_logger.removeHandler(handler)


def test_suppress_logging_restores_level() -> None:
    test_logger = logging.getLogger("test_decoraters2_restore")
    test_logger.setLevel(logging.WARNING)

    with decoraters2.suppress_logging("test_decoraters2_restore"):
        pass

    assert test_logger.getEffectiveLevel() == logging.WARNING


def test_suppress_logging_custom_level() -> None:
    handler = _ListHandler()
    test_logger = logging.getLogger("test_decoraters2_custom")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    with decoraters2.suppress_logging("test_decoraters2_custom", logging.CRITICAL + 1):
        test_logger.critical("hidden too")

    assert len(handler.records) == 0
    test_logger.removeHandler(handler)


def test_suppress_logging_restores_on_exception() -> None:
    test_logger = logging.getLogger("test_decoraters2_exc")
    test_logger.setLevel(logging.INFO)

    with pytest.raises(RuntimeError):
        with decoraters2.suppress_logging("test_decoraters2_exc"):
            raise RuntimeError("oops")

    assert test_logger.getEffectiveLevel() == logging.INFO


def test_suppress_logging_none_name_targets_root() -> None:
    root = logging.getLogger()
    original = root.getEffectiveLevel()

    with decoraters2.suppress_logging():
        assert root.getEffectiveLevel() == logging.CRITICAL

    assert root.getEffectiveLevel() == original


# ---------------------------------------------------------------------------
# on_error
# ---------------------------------------------------------------------------


def test_on_error_catches_and_reraises() -> None:
    ex = SQLAlchemyError("db fail")
    with pytest.raises(SQLAlchemyError):
        decoraters2.on_error(ex)


def test_on_error_catches_without_reraise() -> None:
    ex = SQLAlchemyError("db fail")
    decoraters2.on_error(ex, reraise=None)


def test_on_error_raises_uncaught_exception() -> None:
    ex = ValueError("not a db error")
    with pytest.raises(ValueError, match="not a db error"):
        decoraters2.on_error(ex)


def test_on_error_logs_exception_attr() -> None:
    ex = SQLAlchemyError("db fail")
    ex.exception = "inner detail"
    with pytest.raises(SQLAlchemyError):
        decoraters2.on_error(ex)


def test_on_error_custom_catches() -> None:
    ex = KeyError("missing")
    with pytest.raises(SQLAlchemyError):
        decoraters2.on_error(ex, catches=(KeyError,), reraise=SQLAlchemyError)


def test_on_error_custom_catches_not_matched() -> None:
    ex = ValueError("wrong type")
    with pytest.raises(ValueError, match="wrong type"):
        decoraters2.on_error(ex, catches=(KeyError,))


# ---------------------------------------------------------------------------
# transaction
# ---------------------------------------------------------------------------


def test_transaction_commit(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> int:
        return 42

    result = func()
    assert result == 42
    db.session.commit.assert_called_once()


def test_transaction_rollback(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        func()

    db.session.commit.assert_not_called()
    db.session.rollback.assert_called_once()


def test_transaction_nested(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def inner() -> int:
        return 1

    @decoraters2.transaction()
    def outer() -> int:
        inner()
        raise ValueError("outer fail")

    with pytest.raises(ValueError, match="outer fail"):
        outer()

    db.session.commit.assert_not_called()
    db.session.rollback.assert_called_once()


def test_transaction_on_error_callback(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")
    handler = Mock(return_value="handled")

    @decoraters2.transaction(on_error=handler)
    def func() -> None:
        raise RuntimeError("boom")

    result = func()
    assert result == "handled"
    handler.assert_called_once()
    db.session.rollback.assert_called_once()


def test_transaction_no_error_handler(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction(on_error=None)
    def func() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        func()
    db.session.rollback.assert_called_once()


def test_transaction_resets_in_transaction_flag(mocker: MockerFixture) -> None:
    mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> int:
        return 1

    func()

    from flask import g

    assert not getattr(g, "in_transaction", False)
