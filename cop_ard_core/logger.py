# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Logging infrastructure: structlog + standard library logging configuration
from YAML, and a timeit decorator for execution-duration metrics.
"""

import logging
import sys
import time as tt
from functools import wraps
from pathlib import Path
from typing import Any

import structlog
import yaml
from structlog.dev import RichTracebackFormatter, plain_traceback


def configure_logger(config_path: str | Path | None = None) -> None:
    """
    Configures structlog and standard Python logging pipelines using a YAML configuration file.

    This function sets up a unified logging infrastructure that intercepts both native
    structlog events and standard library logs (e.g., from third-party packages like EOPF or Dask).
    It automatically detects whether stdout is a terminal (TTY) or redirected to a file,
    adjusting ANSI color escape sequences and traceback formatting accordingly to ensure
    clean, readable output in log files.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If None or invalid, it falls back to the
        default 'logging_config.yaml' located in the cop_ard_core package directory.
    """
    # Resolve to package-relative default config file if no path is explicitly provided
    if config_path is None:
        config_path = Path(__file__).parent / "logging_config.yaml"

    path = Path(config_path)

    # Baseline fallback configuration parameters in case YAML is missing key-values
    config = {
        "log_level": "INFO",
        "json_format": False,
        "show_timestamps": True,
        "show_caller_info": True,
        "console_colors": True,
        "third_party_levels": {},
    }

    # Load and merge configuration parameters if the YAML file exists
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f)
            if isinstance(loaded_config, dict):
                config.update(loaded_config)

    # Convert string log level (e.g., "INFO") into logging module constant (e.g., logging.INFO)
    log_level = getattr(logging, str(config["log_level"]).upper(), logging.INFO)

    # Detect interactive TTY output stream (True when printing to terminal, False when redirected to .log file)
    is_tty = sys.stdout.isatty()
    use_colors = bool(config.get("console_colors", True)) and is_tty

    # 1. Processors for foreign log records (e.g., standard library calls from EOPF/Dask)
    foreign_pre_chain: Any = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]
    if config.get("show_timestamps"):
        foreign_pre_chain.append(structlog.processors.TimeStamper(fmt="iso"))

    # 2. Processors for native structlog calls across cop_ard_* packages
    structlog_processors: Any = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if config.get("show_timestamps"):
        structlog_processors.append(structlog.processors.TimeStamper(fmt="iso"))

    # Inject caller metadata (filename and line number) while ignoring this logger utility file
    if config.get("show_caller_info"):
        structlog_processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                },
                additional_ignores=["cop_ard_core.logger"],  # Bypass logger.py to capture true source callsite
            )
        )

    # Hand off processed structlog events to standard library ProcessorFormatter
    structlog_processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    # Configure global structlog behavior and integration with standard library logging
    structlog.configure(
        processors=structlog_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Dynamically select exception traceback formatter:
    # Use RichTracebackFormatter (with rich box graphics) for interactive terminal sessions,
    # and plain_traceback (clean standard text) when output is saved to file redirection.
    if is_tty:
        exc_formatter = RichTracebackFormatter(
            show_locals=False,
            max_frames=10,
        )
    else:
        exc_formatter = plain_traceback

    # Select output renderer (JSON for production environments, ConsoleRenderer for human-readable output)
    renderer = (
        structlog.processors.JSONRenderer()
        if config.get("json_format")
        else structlog.dev.ConsoleRenderer(
            colors=use_colors,
            exception_formatter=exc_formatter,
            pad_event=is_tty,  # Disable padding whitespace when writing to log files to avoid trailing spaces
        )
    )

    # Create a unified ProcessorFormatter combining native structlog and foreign log chains
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=foreign_pre_chain,
    )

    # Reset existing root logger handlers to prevent duplicate log outputs
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Attach configured StreamHandler outputting to standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(log_level)

    # Configure specific log levels for third-party libraries (e.g., EOPF, Dask, urllib3)
    third_party = config.get("third_party_levels", {})
    third_party = third_party if isinstance(third_party, dict) else {}
    if "eopf" not in third_party:
        third_party["eopf"] = config["log_level"]

    for lib_name, level_str in third_party.items():
        lib_level = getattr(logging, str(level_str).upper(), log_level)
        logging.getLogger(lib_name).setLevel(lib_level)


# Perform default initial setup upon importing the logger module
configure_logger()
logger = structlog.get_logger(__name__)


def timeit(func):
    """
    Decorator measuring and logging the execution duration of decorated functions.

    Utilizes high-precision monotonic timing (`time.perf_counter`) and logs metrics
    as structured key-value parameters using structlog.

    Parameters
    ----------
    func : callable
        The function or method to be monitored.

    Returns
    -------
    callable
        The wrapped function that measures execution time upon invocation.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        """
        Run the wrapped function and log its execution duration via structlog.
        """
        start_time = tt.perf_counter()
        result = func(*args, **kwargs)
        total_time = tt.perf_counter() - start_time

        # Format duration string dynamically for optimal human readability (milliseconds vs seconds)
        duration_str = f"{total_time * 1000:.2f}ms" if total_time < 1.0 else f"{total_time:.2f}s"

        # Log execution event (callsite metadata is accurately resolved via additional_ignores filter)
        logger.info(
            f"Executed '{func.__qualname__}'",
            elapsed=duration_str,
            duration_sec=round(total_time, 3),
        )
        return result

    return timeit_wrapper
