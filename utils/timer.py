from utils.logger import Logger
from time import perf_counter_ns

class Timer:
    def __init__(self, label=None):
        self.label = f"'{label}' " if label else ""
        self.start_ns = None

    def __enter__(self):
        Logger.info(f"Entering block {self.label}...")
        self.start_ns = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ns = perf_counter_ns()
        duration_ns = end_ns - self.start_ns
        duration_sec = duration_ns / 1e9
        Logger.info(f"Block {self.label}took {duration_sec:.9f} seconds")

class timedmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self.func

        def wrapper(*args, **kwargs):
            Logger.info(f"Calling function '{self.func.__name__}'...")
            start_ns = perf_counter_ns()
            result = self.func(instance, *args, **kwargs)
            end_ns = perf_counter_ns()
            duration_sec = (end_ns - start_ns) / 1e9
            Logger.info(f"Function '{self.func.__name__}' took {duration_sec:.9f} seconds")
            return result

        return wrapper
