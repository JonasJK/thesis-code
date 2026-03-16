"""
Decorators for NIR prediction models.
"""

import cProfile
import functools
import io
import logging
import pstats

log = logging.getLogger(__name__)


def profile_execution(func):
    """
    Decorator that profiles the execution of a function using cProfile.
    Prints the top 10 functions by total time.

    Usage:
        @profile_execution
        def main():
            # your code here
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            pr.disable()

            # Generate profiling report
            s = io.StringIO()
            stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime")
            stats.print_stats(10)

            print("\n" + "=" * 60)
            print("PROFILING RESULTS (Top 10 by Total Time)")
            print("=" * 60)
            print(s.getvalue())
            print("=" * 60)

        return result

    return wrapper


def profile_execution_detailed(num_stats=20, sort_by="tottime"):
    """
    Decorator that profiles the execution of a function with customizable options.

    Parameters:
    -----------
    num_stats : int
        Number of top functions to display (default: 20)
    sort_by : str
        Sorting criteria ('tottime', 'cumtime', 'calls', etc.)

    Usage:
        @profile_execution_detailed(num_stats=15, sort_by='cumtime')
        def main():
            # your code here
            pass
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                pr.disable()

                # Generate profiling report
                s = io.StringIO()
                stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_by)
                stats.print_stats(num_stats)

                print("\n" + "=" * 80)
                print(f"PROFILING RESULTS (Top {num_stats} by {sort_by})")
                print("=" * 80)
                print(s.getvalue())
                print("=" * 80)

            return result

        return wrapper

    return decorator


def timing_decorator(func):
    """
    Simple decorator that just times function execution.

    Usage:
        @timing_decorator
        def main():
            # your code here
            pass
    """
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        finally:
            duration = time.time() - start_time
            print(f"\n⏱️  Function '{func.__name__}' completed in {duration:.2f} seconds")

        return result

    return wrapper
