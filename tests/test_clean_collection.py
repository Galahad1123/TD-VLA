import importlib.util
from pathlib import Path


def runner():
    path = Path(__file__).parents[1] / "scripts/collect_clean_subset.py"
    assert path.exists(), "Batch runner missing"
    spec = importlib.util.spec_from_file_location("clean_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduler_counts_success_not_attempts_and_advances_seed():
    m = runner()
    records = [
        dict(task="a", seed=0, success=False),
        dict(task="a", seed=1, success=True),
        dict(task="b", seed=0, success=True),
    ]
    assert m.next_episode(["a", "b"], records, {"b"}, 2, 10) == ("a", 2)
    assert m.next_episode(["a", "b"], records, {"a", "b"}, 2, 10) is None


def test_scheduler_stops_at_target_and_refuses_exhausted_attempts():
    import pytest

    m = runner()
    records = [dict(task="a", seed=0, success=True)]
    assert m.next_episode(["a"], records, set(), 1, 1) is None
    with pytest.raises(RuntimeError, match="attempt"):
        m.next_episode(["a"], records, set(), 2, 1)


def test_scheduler_keeps_task_identity_and_prefers_least_complete():
    m = runner()
    records = [
        dict(task="a", seed=0, success=True),
        dict(task="a", seed=1, success=True),
        dict(task="b", seed=4, success=False),
    ]
    assert m.next_episode(["a", "b"], records, set(), 3, 10) == ("b", 5)


def test_lock_belongs_to_dataset_not_logs(tmp_path):
    import pytest

    m = runner()
    first = m.acquire_dataset_lock(tmp_path)
    try:
        with pytest.raises(BlockingIOError):
            m.acquire_dataset_lock(tmp_path)
    finally:
        first.close()
    m.acquire_dataset_lock(tmp_path).close()


def test_interrupted_annotation_is_not_published_and_can_resume(tmp_path):
    import pytest

    m = runner()
    final = tmp_path / "seed-0000"

    def fail(path, output, offsets):
        output.mkdir()
        (output / "report.json").write_text("{}")
        raise RuntimeError("interrupted before review image")

    with pytest.raises(RuntimeError):
        m.annotate_atomic(None, final, [], fail)
    assert not final.exists()

    def succeed(path, output, offsets):
        output.mkdir()
        (output / "report.json").write_text("{}")
        (output / "review.jpg").write_bytes(b"review")
        return {"status": "ok"}

    assert m.annotate_atomic(None, final, [], succeed) == {"status": "ok"}
    assert (final / "review.jpg").exists()
    assert not final.with_name(final.name + ".partial").exists()


def test_two_workers_per_gpu_have_distinct_slots():
    m = runner()
    assert m.worker_slots(["0", "1"], 2) == [("0", 0), ("1", 0), ("0", 1), ("1", 1)]


def test_validation_budget_prevents_extra_attempts_per_task():
    m = runner()
    records = [dict(task="a", seed=0, success=False), dict(task="b", seed=0, success=True)]
    assert m.next_episode(
        ["a", "b"], records, set(), 50, 500, submitted={"a": 4, "b": 3}, attempts_per_task=4
    ) == ("b", 1)
    assert (
        m.next_episode(
            ["a", "b"], records, set(), 50, 500, submitted={"a": 4, "b": 4}, attempts_per_task=4
        )
        is None
    )
