# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time as tt
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)


# Function to monitor execution time
def timeit(func):
    """
    A decorator that logs the execution time of a function.

    This decorator can be applied to any function to measure and log how long
    it takes to execute, from the moment it is called until it returns. It uses
    `time.perf_counter` for an accurate measurement of wall-clock time.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function, which when called, will execute the original
        function and log its execution time before returning the result.
    """

    # Use @wraps to preserve the original function's metadata (like name and docstring).
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        # Record the starting time before function execution.
        start_time = tt.perf_counter()
        # Call the original decorated function with all its arguments.
        result = func(*args, **kwargs)
        # Record the ending time after the function has completed.
        end_time = tt.perf_counter()
        # Calculate the total elapsed time.
        total_time = end_time - start_time
        # Log the function name and its total execution time.
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        # Return the original result of the decorated function.
        return result

    # Return the wrapper function, which replaces the original function.
    return timeit_wrapper
