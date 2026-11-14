"""Resume six-task Clean collection with one child per GPU and v4 annotation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

TASKS = [
    "move_stapler_pad",
    "lift_pot",
    "handover_block",
    "beat_block_hammer",
    "stack_blocks_two",
    "place_object_basket",
]
ROOT = Path(__file__).resolve().parents[1]


def next_episode(tasks, records, busy, target, max_attempts):
    candidates = []
    for task in tasks:
        rows = [r for r in records if r["task"] == task]
        successes = sum(bool(r["success"]) for r in rows)
        if successes >= target or task in busy:
            continue
        if len(rows) >= max_attempts:
            raise RuntimeError(f"{task}: attempt limit reached")
        candidates.append(
            (successes, tasks.index(task), task, max((r["seed"] for r in rows), default=-1) + 1)
        )
    if not candidates:
        return None
    _, _, task, seed = min(candidates)
    return task, seed


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_episode(task, seed, gpu, data, logs, sim_python, timeout):
    import h5py

    log = logs / f"{task}-{seed:04d}.log"
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
    command = [
        str(sim_python),
        str(ROOT / "scripts/collect_pilot.py"),
        "--task",
        task,
        "--seed",
        str(seed),
        "--output",
        str(data),
    ]
    try:
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
            )
        if result.returncode != 0:
            raise RuntimeError(f"collector exited {result.returncode}")
        path = data / task / f"seed-{seed:04d}.h5"
        with h5py.File(path) as f:
            record = dict(
                task=task,
                seed=seed,
                success=bool(f.attrs["expert_success"]),
                frames=len(f["qpos"]),
                path=str(path.relative_to(data)),
                sha256=digest(path),
            )
        outcome = dict(record=record, elapsed_seconds=time.monotonic() - started, gpu=gpu)
        if not record["success"]:
            outcome["failure_log_tail"] = log.read_text(errors="replace")[-2000:]
        log.unlink()
        return outcome
    except Exception as error:
        return dict(
            task=task,
            seed=seed,
            gpu=gpu,
            elapsed_seconds=time.monotonic() - started,
            error=str(error),
            log=str(log),
        )


def acquire_dataset_lock(data):
    lock = (data.resolve() / ".collection.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        lock.close()
        raise
    return lock


def annotate_atomic(path, directory, offsets, annotate_function):
    staging = directory.with_name(directory.name + ".partial")
    if staging.is_symlink():
        raise RuntimeError(f"Unexpected annotation symlink: {staging}")
    if staging.exists():
        shutil.rmtree(staging)
    result = annotate_function(path, staging, offsets)
    staging.rename(directory)
    return result


def main():
    import h5py
    import numpy as np
    from annotate_pilot import annotate, SCHEMA

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--sim-python", type=Path, default=ROOT / "envs/sim/.venv/bin/python")
    a = p.parse_args()
    a.data, a.labels, a.logs = a.data.resolve(), a.labels.resolve(), a.logs.resolve()
    gpus = a.gpus.split(",")
    if a.target < 1 or a.max_attempts < a.target or a.timeout < 1 or len(set(gpus)) != len(gpus):
        p.error("Invalid target, attempts, timeout or duplicate GPUs")
    a.logs.mkdir(parents=True, exist_ok=True)
    lock = acquire_dataset_lock(a.data)
    a.labels.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (a.data / "manifest.jsonl").read_text().splitlines()]
    keys = [(r["task"], r["seed"]) for r in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate task/seed in raw manifest")
    partials = list(a.data.glob("*/*.partial.h5"))
    if partials:
        raise RuntimeError(f"Inspect incomplete raw files before resuming: {partials}")
    for r in records:
        path = a.data / r["path"]
        if r["task"] not in TASKS or digest(path) != r["sha256"]:
            raise RuntimeError(f"Invalid raw record: {path}")
        with h5py.File(path) as f:
            assert bool(f.attrs["expert_success"]) == r["success"]
    sources = [
        "scripts/collect_pilot.py",
        "scripts/collect_clean_subset.py",
        "scripts/annotate_pilot.py",
        "src/td_vla/annotation/geometry.py",
        "src/td_vla/annotation/stages.py",
        "configs/pilot_geometry.json",
    ]
    source_hashes = {name: digest(ROOT / name) for name in sources}
    provenance = dict(
        target_per_task=a.target,
        max_attempts=a.max_attempts,
        timeout_seconds=a.timeout,
        sim_python=str(a.sim_python.resolve()),
        tasks=TASKS,
        gpus=gpus,
        source_sha256=source_hashes,
        schema=SCHEMA,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    previous = a.logs / "provenance.json"
    if previous.exists() and json.loads(previous.read_text()) != provenance:
        raise RuntimeError("Resume provenance/configuration mismatch")
    save_json(previous, provenance)
    reports = []
    offsets = np.linspace(0.3, 3.0, 10)

    def annotate_record(r):
        if not r["success"]:
            return dict(task=r["task"], seed=r["seed"], status="skipped_failed_expert")
        directory = a.labels / r["task"] / Path(r["path"]).stem
        if directory.exists():
            for filename in ("report.json", "labels.npz", "boundaries.json", "review.jpg"):
                if not (directory / filename).is_file():
                    raise RuntimeError(f"Incomplete published annotation: {directory}")
            report = json.loads((directory / "report.json").read_text())
            if report["raw_sha256"] != r["sha256"] or report["schema"] != SCHEMA:
                raise RuntimeError(f"Stale annotation: {directory}")
            with np.load(directory / "labels.npz") as z:
                assert len(z["stage"]) == r["frames"]
            return report
        return annotate_atomic(a.data / r["path"], directory, offsets, annotate)

    def status(state, running, error=None):
        save_json(
            a.logs / "status.json",
            dict(
                state=state,
                pid=os.getpid(),
                updated_utc=datetime.now(timezone.utc).isoformat(),
                target_per_task=a.target,
                counts={t: sum(r["success"] for r in records if r["task"] == t) for t in TASKS},
                attempts=len(records),
                labelled=sum(r["status"] == "ok" for r in reports),
                review_required=sum(r["status"] == "review_required" for r in reports),
                running=running,
                error=error,
            ),
        )

    status("initializing", [])
    try:
        reports.extend(annotate_record(r) for r in records)
        failure = None
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            active = {}
            while True:
                if not failure:
                    for gpu in gpus:
                        if any(job["gpu"] == gpu for job in active.values()):
                            continue
                        try:
                            item = next_episode(
                                TASKS,
                                records,
                                {j["task"] for j in active.values()},
                                a.target,
                                a.max_attempts,
                            )
                        except RuntimeError as error:
                            failure = str(error)
                            break
                        if item:
                            task, seed = item
                            future = pool.submit(
                                run_episode,
                                task,
                                seed,
                                gpu,
                                a.data,
                                a.logs,
                                a.sim_python,
                                a.timeout,
                            )
                            active[future] = dict(task=task, seed=seed, gpu=gpu)
                status("stopping" if failure else "running", list(active.values()), failure)
                if not active:
                    break
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    outcome = future.result()
                    with (a.logs / "attempts.jsonl").open("a") as stream:
                        stream.write(json.dumps(outcome) + "\n")
                    if "error" in outcome:
                        failure = outcome["error"]
                    else:
                        record = outcome["record"]
                        records.append(record)
                        reports.append(annotate_record(record))
        if failure:
            status("blocked", [], failure)
            raise RuntimeError(failure)
        if any(sum(r["success"] for r in records if r["task"] == t) != a.target for t in TASKS):
            raise RuntimeError("Final success counts differ from requested target")
        save_json(
            a.labels / "manifest.json",
            dict(
                schema=SCHEMA,
                source_sha256=source_hashes,
                sampling="nearest_timestamp_ties_earlier",
                reports=reports,
            ),
        )
        status("complete", [])
        lock.close()
    except Exception as error:
        status("blocked", [], str(error))
        raise


if __name__ == "__main__":
    main()
