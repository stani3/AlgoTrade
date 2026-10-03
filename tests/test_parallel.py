import os

import pytest

from algotrade import parallel
from algotrade.parallel import ENV, THREAD_VARS, Pool, default_workers, resolve_workers, run


def pid_and_square(value: int) -> tuple[int, int]:
    return os.getpid(), value * value


def thread_settings(_: object) -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_VARS}


def fail_on_three(value: int) -> int:
    if value == 3:
        raise ValueError("three")
    return value


@pytest.fixture
def jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker processes need numba's JIT on (coverage runs switch it off)."""

    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)


def test_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setenv(ENV, "6")
    assert default_workers() == 6
    assert resolve_workers(None) == 6
    assert resolve_workers(3) == 3
    assert resolve_workers(0) == 1
    monkeypatch.setenv(ENV, "0")
    assert default_workers() == 1
    monkeypatch.delenv(ENV)
    assert default_workers() == max((os.cpu_count() or 1) - 2, 1)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)
    assert default_workers() == 1


def test_coverage_runs_stay_in_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    assert resolve_workers(8) == 1
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")
    assert resolve_workers(8) == 8


def test_one_worker_runs_inline() -> None:
    with Pool(1) as pool:
        found = pool.map(pid_and_square, [1, 2, 3])
    assert found == [(os.getpid(), 1), (os.getpid(), 4), (os.getpid(), 9)]
    assert run(pid_and_square, [5]) == [(os.getpid(), 25)]


def test_a_single_job_runs_inline_even_with_workers(jit: None) -> None:
    with Pool(4) as pool:
        assert pool.map(pid_and_square, [7]) == [(os.getpid(), 49)]
        assert pool._executor is None


def test_workers_return_results_in_order(jit: None) -> None:
    before = {name: os.environ.get(name) for name in THREAD_VARS}
    with Pool(2) as pool:
        found = run(pid_and_square, range(8), pool)
        settings = pool.map(thread_settings, range(2))
    assert [square for _, square in found] == [n * n for n in range(8)]
    assert os.getpid() not in {pid for pid, _ in found}
    assert all(value == "1" for s in settings for value in s.values())
    assert {name: os.environ.get(name) for name in THREAD_VARS} == before


def test_parent_thread_settings_are_restored(jit: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    pool = Pool(2)
    pool.map(pid_and_square, range(3))
    assert os.environ["OMP_NUM_THREADS"] == "1"
    pool.close()
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "MKL_NUM_THREADS" not in os.environ
    pool.close()  # closing twice is harmless


def test_job_errors_reach_the_caller(jit: None) -> None:
    with Pool(2) as pool, pytest.raises(ValueError, match="three"):
        pool.map(fail_on_three, range(5))
    with pytest.raises(ValueError, match="three"):
        run(fail_on_three, range(5))
