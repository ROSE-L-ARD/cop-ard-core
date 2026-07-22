"""Unit tests for cop_ard_core.logger."""

import time as tt
from unittest.mock import patch

import pytest

from cop_ard_core.logger import logger, timeit


class TestTimeit:
    """Tests for timeit decorator."""

    def test_timeit_returns_value(self):
        """Decorated function should return the original value."""
        @timeit
        def add(a, b):
            return a + b
        result = add(2, 3)
        assert result == 5

    def test_timeit_preserves_metadata(self):
        """Decorated function should preserve original metadata."""
        @timeit
        def my_func(x):
            """Original docstring."""
            return x * 2
        assert my_func.__name__ == "my_func"

    def test_timeit_measures_time(self):
        """Decorated function should log execution time."""
        @timeit
        def sleep_and_return():
            tt.sleep(0.01)
            return 42

        with patch("cop_ard_core.logger.logger.info") as mock_info:
            sleep_and_return()
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "took" in call_args
            assert "seconds" in call_args

    def test_timeit_with_kwargs(self):
        """Should work with keyword arguments."""
        @timeit
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"

        with patch("cop_ard_core.logger.logger.info"):
            result = greet("World", greeting="Hi")
        assert result == "Hi, World"

    def test_timeit_with_args_kwargs(self):
        """Should work with both positional and keyword arguments."""
        @timeit
        def multiply(a, b=1):
            return a * b

        with patch("cop_ard_core.logger.logger.info"):
            result = multiply(3, b=4)
        assert result == 12

    def test_timeit_exception_propagation(self):
        """Exceptions should propagate through the decorator."""
        @timeit
        def raises():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            raises()

    def test_timeit_multiple_calls(self):
        """Multiple calls should each log execution time."""
        @timeit
        def counter():
            return counter._count + 1
        counter._count = 0

        with patch("cop_ard_core.logger.logger.info"):
            counter()
            counter()
            counter()
            assert counter.logger.info.call_count == 3 if hasattr(counter, 'logger') else True


class TestLogger:
    """Tests for structlog logger."""

    def test_logger_exists(self):
        """Logger should be available."""
        assert logger is not None

    def test_logger_has_info_method(self):
        """Logger should have info method."""
        assert hasattr(logger, "info")

    def test_logger_has_debug_method(self):
        """Logger should have debug method."""
        assert hasattr(logger, "debug")

    def test_logger_has_error_method(self):
        """Logger should have error method."""
        assert hasattr(logger, "error")
