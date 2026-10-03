"""Run independent research jobs (one per symbol) in worker processes.

Backtests are CPU-bound Python and numpy, so threads would queue on the interpreter lock; the
jobs run in separate processes instead. Each job gets one symbol's bars and does all the work on
that symbol, so the bars are sent to a worker once. Results come back in the order the jobs were
given, which keeps every table identical to a serial run.

The worker count comes from ``workers``, else ``ALGOTRADE_WORKERS``, else the number of CPUs
minus two. One worker runs the jobs inline in this process, as do coverage runs
(``NUMBA_DISABLE_JIT=1``), whose measurements would otherwise be lost in the workers.

Job functions must be importable module-level functions: Windows starts workers by importing
afresh ("spawn"), not by copying this process.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Self

ENV = "ALGOTRADE_WORKERS"

# Every worker gets one thread for numpy's maths libraries and numba, or 18 workers would each
# start a thread per CPU. Workers read these when they start, so they are set before they do.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def default_workers() -> int:
    value = os.environ.get(ENV, "").strip()
    if value:
        return max(int(value), 1)
    return max((os.cpu_count() or 1) - 2, 1)


def resolve_workers(workers: int | None) -> int:
    if os.environ.get("NUMBA_DISABLE_JIT", "0") not in ("", "0"):
        return 1
    return max(int(workers), 1) if workers is not None else default_workers()


def _warm_numba() -> None:
    """Compile (or load from cache) the shared numba kernels here, before workers need them."""

    import numpy as np

    from algotrade.backtest.bracket import _simulate
    from algotrade.indicators import entry_exit_state

    one, flag = np.ones(2), np.zeros(2, dtype=bool)
    _simulate(one, one, one, one, one * 0, flag, flag, one, one, 0, 0, 1.0, 0.0, 0.0, 1.0, 0)
    entry_exit_state(flag, flag, flag, flag)


class Pool:
    """``with Pool(workers) as pool: pool.map(job, items)``; inline when one worker is enough."""

    def __init__(self, workers: int | None = None) -> None:
        self.workers = resolve_workers(workers)
        self._executor: ProcessPoolExecutor | None = None
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved = {}

    def _start(self) -> ProcessPoolExecutor:
        if self._executor is None:
            _warm_numba()
            self._saved = {name: os.environ.get(name) for name in THREAD_VARS}
            os.environ.update({name: "1" for name in THREAD_VARS})
            self._executor = ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"))
        return self._executor

    def map(self, job: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        items = list(items)
        if self.workers == 1 or len(items) <= 1:
            return [job(item) for item in items]
        return list(self._start().map(job, items))


def run(job: Callable[[Any], Any], items: Iterable[Any], pool: Pool | None = None) -> list[Any]:
    """``pool.map`` when a pool is given, else the jobs one after another in this process."""

    return pool.map(job, items) if pool is not None else [job(item) for item in items]
