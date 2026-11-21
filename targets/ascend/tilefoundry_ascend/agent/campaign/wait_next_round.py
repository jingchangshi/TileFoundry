import time


def check_stalls(
    last_progress_time: float,
    start_time: float,
    timeout_idle: int = 120,
    timeout_total: int = 900,
) -> str | None:
    """Differentiate 'Idle-Stall' (120s) and 'Progress-Total' (900s)."""
    now = time.time()
    if now - last_progress_time > timeout_idle:
        return "Idle-Stall"
    if now - start_time > timeout_total:
        return "Progress-Total"
    return None
