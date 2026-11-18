import time
from targets.ascend.tilefoundry_ascend.agent.campaign.wait_next_round import check_stalls

def test_stalls():
    now = time.time()
    # Idle-Stall (120s)
    # Progress-Total (900s)
    assert check_stalls(now - 100, now) is None
    assert check_stalls(now - 130, now) == "Idle-Stall"
    assert check_stalls(now - 10, now - 910) == "Progress-Total"
