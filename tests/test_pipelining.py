import threading
import time

import pytest

from govscape.pipelining import StreamStage, run_pipelined


def test_items_pass_through_stages_in_order():
    seen = []
    run_pipelined(range(5), [lambda x: x * 10, lambda x: x + 1, seen.append])
    assert seen == [1, 11, 21, 31, 41]


def test_stages_overlap():
    running = set()
    overlapped = threading.Event()

    def stage(name):
        def run(item):
            running.add(name)
            if len(running) > 1:
                overlapped.set()
            time.sleep(0.2)
            running.discard(name)
            return item

        return run

    run_pipelined(range(3), [stage("a"), stage("b")])
    assert overlapped.is_set()


def test_error_in_a_stage_stops_the_pipeline_and_is_raised():
    produced = []

    def items():
        for i in range(1000):
            produced.append(i)
            yield i

    def fail_on_two(x):
        if x == 2:
            raise ValueError("bad item")
        return x

    with pytest.raises(ValueError, match="bad item"):
        run_pipelined(items(), [fail_on_two, lambda x: x])
    # The source stops soon after the failure instead of producing everything.
    assert len(produced) < 10


def test_a_stream_stage_can_regroup_items():
    def pairs(items):
        pair = []
        for item in items:
            pair.append(item)
            if len(pair) == 2:
                yield pair
                pair = []
        if pair:
            yield pair

    seen = []
    run_pipelined(range(5), [lambda x: x * 10, StreamStage(pairs), seen.append])
    assert seen == [[0, 10], [20, 30], [40]]
