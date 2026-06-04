import time
import uuid
from contextlib import contextmanager


def generate_request_id() -> str:
    """
    Generate unique request ID
    """
    return f"req_{uuid.uuid4().hex[:12]}"


@contextmanager
def trace_execution():
    """
    Measure execution time
    """

    start_time = time.perf_counter()

    try:
        yield

    finally:
        end_time = time.perf_counter()
        execution_time = round(
            (end_time - start_time) * 1000,
            2
        )

        print(
            f"Execution time: {execution_time} ms"
        )