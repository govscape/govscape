import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

_DONE = object()


@dataclass
class StreamStage:
    """A pipeline stage that maps its whole input stream to an output stream,
    e.g. to regroup items, rather than each input item to one output."""

    fn: Callable[[Iterator], Iterable]


def run_pipelined(
    items: Iterable, stages: list[Callable | StreamStage], depth: int = 1
) -> None:
    """Pass each item through stages in order, each stage in its own thread, so
    stage i works on item k while stage i + 1 works on item k - 1. Each stage's
    return value is the next stage's input; a StreamStage instead yields its
    outputs as it consumes its inputs. Items are produced by iterating `items` in
    its own thread too, and at most `depth` results wait between neighbouring
    stages. The first exception stops every stage and is re-raised.
    """
    queues: list[queue.Queue] = [queue.Queue(maxsize=depth) for _ in stages]
    stop = threading.Event()
    errors: list[BaseException] = []

    def put(q: queue.Queue, item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def get(q: queue.Queue):
        while not stop.is_set():
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                pass
        return _DONE

    def fail(error: BaseException) -> None:
        errors.append(error)
        stop.set()

    def produce() -> None:
        try:
            for item in items:
                if not put(queues[0], item):
                    return
            put(queues[0], _DONE)
        except BaseException as error:
            fail(error)

    def inputs(i: int) -> Iterator:
        while (item := get(queues[i])) is not _DONE:
            yield item

    def work(i: int) -> None:
        stage = stages[i]
        try:
            if isinstance(stage, StreamStage):
                results = iter(stage.fn(inputs(i)))
            else:
                results = (stage(item) for item in inputs(i))
            for result in results:
                if i + 1 < len(stages) and not put(queues[i + 1], result):
                    return
            if i + 1 < len(stages):
                put(queues[i + 1], _DONE)
        except BaseException as error:
            fail(error)

    threads = [threading.Thread(target=produce, name="pipeline-source")]
    threads += [
        threading.Thread(target=work, args=(i,), name=f"pipeline-stage-{i}")
        for i in range(len(stages))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
