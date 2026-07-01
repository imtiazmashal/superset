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
from superset.utils.backports import StrEnum

# ---------------------------------------------------------------------------
# statsd_gauge
# ---------------------------------------------------------------------------


class ResponseValues(StrEnum):
    FAIL = "fail"
    WARN = "warn"
    OK = "ok"


@pytest.mark.parametrize(
    "response_value, expected_exception, expected_result",
    [
        (ResponseValues.OK, None, "custom.prefix.ok"),
        (ResponseValues.FAIL, ValueError, "custom.prefix.error"),
        (ResponseValues.WARN, FileNotFoundError, "custom.prefix.warning"),
    ],
)
def test_statsd_gauge_with_custom_prefix(
    response_value: str,
    expected_exception: Optional[type[Exception]],
    expected_result: str,
) -> None:
    @decoraters2.statsd_gauge("custom.prefix")
    def my_func(response: ResponseValues) -> str:
        if response == ResponseValues.FAIL:
            raise ValueError("Error")
        if response == ResponseValues.WARN:
            exc = FileNotFoundError("Not found")
            exc.status = 404  # type: ignore[attr-defined]
            raise exc
        return "OK"

    with patch.dict("flask.current_app.config", {"STATS_LOGGER": Mock()}):
        from flask import current_app

        cm = (
            pytest.raises(expected_exception)
            if isclass(expected_exception) and issubclass(expected_exception, Exception)
            else nullcontext()
        )

        with cm:
            my_func(response_value)

        current_app.config["STATS_LOGGER"].gauge.assert_called_once_with(
            expected_result, 1
        )


def test_statsd_gauge_uses_function_name_when_no_prefix() -> None:
    @decoraters2.statsd_gauge()
    def some_function() -> str:
        return "OK"

    with patch.dict("flask.current_app.config", {"STATS_LOGGER": Mock()}):
        from flask import current_app

        some_function()
        current_app.config["STATS_LOGGER"].gauge.assert_called_once_with(
            "some_function.ok", 1
        )


def test_statsd_gauge_error_without_status_attribute() -> None:
    @decoraters2.statsd_gauge("prefix")
    def my_func() -> None:
        raise RuntimeError("boom")

    with patch.dict("flask.current_app.config", {"STATS_LOGGER": Mock()}):
        from flask import current_app

        with pytest.raises(RuntimeError, match="boom"):
            my_func()

        current_app.config["STATS_LOGGER"].gauge.assert_called_once_with(
            "prefix.error", 1
        )


def test_statsd_gauge_error_with_status_500() -> None:
    @decoraters2.statsd_gauge("prefix")
    def my_func() -> None:
        exc = Exception("server error")
        exc.status = 500  # type: ignore[attr-defined]
        raise exc

    with patch.dict("flask.current_app.config", {"STATS_LOGGER": Mock()}):
        from flask import current_app

        with pytest.raises(Exception, match="server error"):
            my_func()

        current_app.config["STATS_LOGGER"].gauge.assert_called_once_with(
            "prefix.error", 1
        )


# ---------------------------------------------------------------------------
# logs_context
# ---------------------------------------------------------------------------


@patch("superset.utils.decoraters2.g")
def test_logs_context_no_kwargs(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(1, 2)
    assert flask_g_mock.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_from_function_kwargs(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(dashboard_id=5, slice_id=10)
    assert flask_g_mock.logs_context == {"dashboard_id": 5, "slice_id": 10}


@patch("superset.utils.decoraters2.g")
def test_logs_context_filters_disallowed_keys(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(slice_id=1, bad_key="nope")
    assert flask_g_mock.logs_context == {"slice_id": 1}


@patch("superset.utils.decoraters2.g")
def test_logs_context_decorator_kwargs_override_function_kwargs(
    flask_g_mock: Mock,
) -> None:
    flask_g_mock.logs_context = {}
    uid = uuid.uuid4()

    @decoraters2.logs_context(slice_id=99, execution_id=uid)
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(slice_id=1)
    assert flask_g_mock.logs_context["slice_id"] == 99
    assert flask_g_mock.logs_context["execution_id"] == uid


@patch("superset.utils.decoraters2.g")
def test_logs_context_with_context_func(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context(
        context_func=lambda *args, **kwargs: {"dashboard_id": kwargs.get("did")}
    )
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(did=42)
    assert flask_g_mock.logs_context == {"dashboard_id": 42}


@patch("superset.utils.decoraters2.g")
def test_logs_context_context_func_overrides_both(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context(
        slice_id=10,
        context_func=lambda *args, **kwargs: {"slice_id": 20},
    )
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(slice_id=1)
    assert flask_g_mock.logs_context == {"slice_id": 20}


@patch("superset.utils.decoraters2.g")
def test_logs_context_skips_none_values(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(slice_id=None, dashboard_id=5)
    assert flask_g_mock.logs_context == {"dashboard_id": 5}


@patch("superset.utils.decoraters2.g")
def test_logs_context_initializes_g_logs_context(flask_g_mock: Mock) -> None:
    del flask_g_mock.logs_context  # simulate missing attribute

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(slice_id=1)
    assert flask_g_mock.logs_context == {"slice_id": 1}


@patch("superset.utils.decoraters2.g")
def test_logs_context_bad_context_func_does_not_crash(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context(context_func=lambda: "not a dict")  # type: ignore
    def myfunc() -> str:
        return "test"

    result = myfunc()
    assert result == "test"
    assert flask_g_mock.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_non_callable_context_func(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context(context_func="not callable")  # type: ignore
    def myfunc() -> str:
        return "test"

    result = myfunc()
    assert result == "test"
    assert flask_g_mock.logs_context == {}


@patch("superset.utils.decoraters2.g")
def test_logs_context_all_allowed_keys(flask_g_mock: Mock) -> None:
    flask_g_mock.logs_context = {}

    @decoraters2.logs_context()
    def myfunc(*args: Any, **kwargs: Any) -> str:
        return "test"

    myfunc(
        slice_id=1,
        dashboard_id=2,
        dataset_id=3,
        execution_id=4,
        report_schedule_id=5,
        extra_key=6,
    )
    assert flask_g_mock.logs_context == {
        "slice_id": 1,
        "dashboard_id": 2,
        "dataset_id": 3,
        "execution_id": 4,
        "report_schedule_id": 5,
    }


# ---------------------------------------------------------------------------
# stats_timing
# ---------------------------------------------------------------------------


def test_stats_timing_calls_timing_on_success() -> None:
    mock_logger = Mock()
    with patch("superset.utils.decoraters2.now_as_float", side_effect=[100.0, 105.0]):
        with decoraters2.stats_timing("my.key", mock_logger) as start:
            assert start == 100.0

    mock_logger.timing.assert_called_once_with("my.key", 5.0)


def test_stats_timing_calls_timing_on_exception() -> None:
    mock_logger = Mock()
    with patch("superset.utils.decoraters2.now_as_float", side_effect=[100.0, 102.5]):
        with pytest.raises(ValueError, match="boom"):
            with decoraters2.stats_timing("err.key", mock_logger):
                raise ValueError("boom")

    mock_logger.timing.assert_called_once_with("err.key", 2.5)


# ---------------------------------------------------------------------------
# arghash
# ---------------------------------------------------------------------------


def test_arghash_same_args_same_hash() -> None:
    h1 = decoraters2.arghash((1, 2), {"a": 3})
    h2 = decoraters2.arghash((1, 2), {"a": 3})
    assert h1 == h2


def test_arghash_different_args_different_hash() -> None:
    h1 = decoraters2.arghash((1,), {"a": 1})
    h2 = decoraters2.arghash((2,), {"a": 1})
    assert h1 != h2


def test_arghash_kwarg_order_does_not_matter() -> None:
    h1 = decoraters2.arghash((), {"a": 1, "b": 2})
    h2 = decoraters2.arghash((), {"b": 2, "a": 1})
    assert h1 == h2


def test_arghash_empty_inputs() -> None:
    h = decoraters2.arghash((), {})
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# debounce
# ---------------------------------------------------------------------------


def test_debounce_deduplicates_same_args() -> None:
    mock = Mock()

    @decoraters2.debounce()
    def myfunc(x: int, y: int) -> int:
        mock(x, y)
        return x + y

    myfunc(1, 2)
    myfunc(1, 2)
    result = myfunc(1, 2)
    mock.assert_called_once_with(1, 2)
    assert result == 3


def test_debounce_calls_again_after_duration() -> None:
    mock = Mock()

    @decoraters2.debounce(duration=0)
    def myfunc(x: int) -> int:
        mock(x)
        return x

    myfunc(1)
    time.sleep(0.01)
    myfunc(1)
    assert mock.call_count == 2


def test_debounce_calls_again_for_different_args() -> None:
    mock = Mock()

    @decoraters2.debounce()
    def myfunc(x: int) -> int:
        mock(x)
        return x

    myfunc(1)
    myfunc(2)
    assert mock.call_count == 2


def test_debounce_kwarg_order_independent() -> None:
    mock = Mock()

    @decoraters2.debounce()
    def myfunc(a: int = 0, b: int = 0) -> int:
        mock(a, b)
        return a + b

    myfunc(a=1, b=2)
    result = myfunc(b=2, a=1)
    mock.assert_called_once_with(1, 2)
    assert result == 3


# ---------------------------------------------------------------------------
# on_security_exception
# ---------------------------------------------------------------------------


def test_on_security_exception_returns_403() -> None:
    mock_self = Mock()
    mock_self.response.return_value = "forbidden"
    ex = Exception("no access")

    with patch(
        "superset.utils.decoraters2.utils.error_msg_from_exception",
        return_value="no access",
    ):
        result = decoraters2.on_security_exception(mock_self, ex)

    assert result == "forbidden"
    mock_self.response.assert_called_once_with(403, message="no access")


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
    test_logger = logging.getLogger("test-suppress-default")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    with decoraters2.suppress_logging("test-suppress-default"):
        test_logger.warning("suppressed")
        test_logger.error("suppressed")
        test_logger.critical("allowed")

    assert len(handler.records) == 1
    assert handler.records[0].levelname == "CRITICAL"


def test_suppress_logging_custom_level() -> None:
    handler = _ListHandler()
    test_logger = logging.getLogger("test-suppress-custom")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    with decoraters2.suppress_logging("test-suppress-custom", logging.CRITICAL + 1):
        test_logger.critical("suppressed")

    assert len(handler.records) == 0


def test_suppress_logging_restores_level() -> None:
    test_logger = logging.getLogger("test-suppress-restore")
    test_logger.setLevel(logging.INFO)

    with decoraters2.suppress_logging("test-suppress-restore"):
        assert test_logger.getEffectiveLevel() == logging.CRITICAL

    assert test_logger.getEffectiveLevel() == logging.INFO


def test_suppress_logging_restores_level_on_exception() -> None:
    test_logger = logging.getLogger("test-suppress-exc")
    test_logger.setLevel(logging.WARNING)

    with pytest.raises(RuntimeError):
        with decoraters2.suppress_logging("test-suppress-exc"):
            raise RuntimeError("boom")

    assert test_logger.getEffectiveLevel() == logging.WARNING


def test_suppress_logging_root_logger() -> None:
    handler = _ListHandler()
    root = logging.getLogger(None)
    original_level = root.getEffectiveLevel()
    root.addHandler(handler)

    try:
        with decoraters2.suppress_logging(None):
            root.error("suppressed")
        assert len(handler.records) == 0
    finally:
        root.setLevel(original_level)
        root.removeHandler(handler)


# ---------------------------------------------------------------------------
# on_error
# ---------------------------------------------------------------------------


def test_on_error_catches_matching_exception_and_reraises() -> None:
    ex = SQLAlchemyError("db fail")
    with pytest.raises(SQLAlchemyError):
        decoraters2.on_error(ex)


def test_on_error_logs_exception_attribute() -> None:
    ex = SQLAlchemyError("db fail")
    ex.exception = "inner error"

    with patch.object(decoraters2.logger, "exception") as mock_log:
        with pytest.raises(SQLAlchemyError):
            decoraters2.on_error(ex)
        mock_log.assert_called_once_with("inner error")


def test_on_error_raises_original_when_not_caught() -> None:
    ex = ValueError("not a db error")
    with pytest.raises(ValueError, match="not a db error"):
        decoraters2.on_error(ex)


def test_on_error_swallows_when_reraise_is_none() -> None:
    ex = SQLAlchemyError("db fail")
    decoraters2.on_error(ex, catches=(SQLAlchemyError,), reraise=None)


def test_on_error_custom_catches_and_reraise() -> None:
    ex = KeyError("missing")
    with pytest.raises(RuntimeError):
        decoraters2.on_error(ex, catches=(KeyError,), reraise=RuntimeError)


def test_on_error_not_in_catches_raises_original() -> None:
    ex = TypeError("type issue")
    with pytest.raises(TypeError, match="type issue"):
        decoraters2.on_error(ex, catches=(ValueError,), reraise=RuntimeError)


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
    db.session.rollback.assert_not_called()


def test_transaction_rollback_with_default_on_error(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> None:
        raise SQLAlchemyError("db error")

    with pytest.raises(SQLAlchemyError):
        func()

    db.session.commit.assert_not_called()
    db.session.rollback.assert_called_once()


def test_transaction_rollback_non_sqlalchemy_error(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> None:
        raise ValueError("bad value")

    with pytest.raises(ValueError, match="bad value"):
        func()

    db.session.commit.assert_not_called()
    db.session.rollback.assert_called_once()


def test_transaction_nested_skips_commit(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def inner() -> int:
        return 10

    @decoraters2.transaction()
    def outer() -> int:
        val = inner()
        return val + 1

    result = outer()
    assert result == 11
    db.session.commit.assert_called_once()


def test_transaction_nested_rollback(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction()
    def inner() -> int:
        return 10

    @decoraters2.transaction()
    def outer() -> None:
        inner()
        raise ValueError("outer error")

    with pytest.raises(ValueError, match="outer error"):
        outer()

    db.session.commit.assert_not_called()
    db.session.rollback.assert_called_once()


def test_transaction_clears_flag_after_success(mocker: MockerFixture) -> None:
    mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> int:
        return 1

    func()
    from flask import g

    assert not getattr(g, "in_transaction", False)


def test_transaction_clears_flag_after_failure(mocker: MockerFixture) -> None:
    mocker.patch("superset.db")

    @decoraters2.transaction()
    def func() -> None:
        raise SQLAlchemyError("fail")

    with pytest.raises(SQLAlchemyError):
        func()

    from flask import g

    assert not getattr(g, "in_transaction", False)


def test_transaction_custom_on_error(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")
    handler = Mock(return_value="handled")

    @decoraters2.transaction(on_error=handler)
    def func() -> None:
        raise ValueError("oops")

    result = func()
    assert result == "handled"
    handler.assert_called_once()
    db.session.rollback.assert_called_once()


def test_transaction_no_on_error_reraises(mocker: MockerFixture) -> None:
    db = mocker.patch("superset.db")

    @decoraters2.transaction(on_error=None)
    def func() -> None:
        raise ValueError("raw")

    with pytest.raises(ValueError, match="raw"):
        func()

    db.session.rollback.assert_called_once()


def test_transaction_preserves_function_metadata(mocker: MockerFixture) -> None:
    mocker.patch("superset.db")

    @decoraters2.transaction()
    def my_documented_func() -> None:
        """Docstring."""

    assert my_documented_func.__name__ == "my_documented_func"
    assert my_documented_func.__doc__ == "Docstring."
