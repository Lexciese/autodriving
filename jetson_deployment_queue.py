#!/usr/bin/env python3
"""Run multiple jetson_deployment_test.py jobs sequentially from a queue file.

Queue file format (CSV or whitespace separated, optional header):

    logdir,route,mode,warmup
    log/ai23_segdep_seq1_..._route_ringroad-711,ringroad-711,model,3
    log/xr20_segdep_seq1_..._route_ugm_baru,ugm_baru,pipeline,3

Multiple routes can be passed to one job with ';', '|', '+' or quoted
whitespace, for example:

    log/ai23_segdep_seq1_..._route_ringroad-711 ringroad-711;ugm_baru model 3

This runs a single job with repeated --route flags, so the routes are combined
into one dataset and produce one aggregated result.

The warmup column is optional. Lines starting with '#' and blank lines are
ignored. A relative logdir is resolved against the queue file directory, the
repo root, and finally <repo_root>/log.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODES = ("model", "pipeline", "both")
DEFAULT_DELAY_S = 20.0

FIELD_ALIASES = {
    "logdir": "logdir",
    "log_dir": "logdir",
    "log": "logdir",
    "route": "route",
    "routes": "route",
    "mode": "mode",
    "scope": "mode",
    "warmup": "warmup",
    "warm_up": "warmup",
}


@dataclass
class Job:
    index: int
    line_no: int
    logdir: Path
    routes: tuple[str, ...]
    mode: str
    warmup: int | None
    raw_logdir: str


def _split_queue_line(line: str) -> list[str]:
    if "," in line:
        return [cell.strip() for cell in next(csv.reader([line]))]
    return shlex.split(line)


def _parse_routes(raw: str) -> tuple[str, ...]:
    parts = re.split(r"[;|+\s]+", raw.strip())
    return tuple(dict.fromkeys(part for part in parts if part))


def _resolve_logdir(raw: str, queue_dir: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [
        queue_dir / path,
        REPO_ROOT / path,
        REPO_ROOT / "log" / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_queue(path: Path) -> list[Job]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read queue file {path}: {exc}") from exc

    rows: list[tuple[int, list[str]]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fields = _split_queue_line(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no}: cannot parse line: {exc}") from exc
        rows.append((line_no, fields))

    if not rows:
        raise SystemExit(f"queue file is empty: {path}")

    header_map: dict[str, int] | None = None
    header_fields = [FIELD_ALIASES.get(cell.strip().lower()) for cell in rows[0][1]]
    if "logdir" in header_fields:
        header_map = {field: i for i, field in enumerate(header_fields) if field is not None}
        rows = rows[1:]

    jobs: list[Job] = []
    for line_no, fields in rows:
        if header_map is not None:
            def get(name: str, default: str = "") -> str:
                pos = header_map.get(name)
                if pos is None or pos >= len(fields):
                    return default
                return fields[pos].strip()
            raw_logdir = get("logdir")
            routes_raw = get("route")
            mode = get("mode")
            warmup_raw = get("warmup")
        else:
            raw_logdir = fields[0].strip() if len(fields) > 0 else ""
            routes_raw = fields[1].strip() if len(fields) > 1 else ""
            mode = fields[2].strip() if len(fields) > 2 else ""
            warmup_raw = fields[3].strip() if len(fields) > 3 else ""

        if not raw_logdir:
            raise SystemExit(f"{path}:{line_no}: missing logdir")
        routes = _parse_routes(routes_raw)
        if not routes:
            raise SystemExit(f"{path}:{line_no}: missing route")
        if not mode:
            raise SystemExit(f"{path}:{line_no}: missing mode")

        mode = mode.lower()
        if mode not in MODES:
            raise SystemExit(
                f"{path}:{line_no}: invalid mode {mode!r}, expected one of {', '.join(MODES)}"
            )

        warmup: int | None = None
        if warmup_raw:
            try:
                warmup = int(warmup_raw)
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid warmup {warmup_raw!r}") from exc
            if warmup < 0:
                raise SystemExit(f"{path}:{line_no}: warmup must be >= 0")

        jobs.append(Job(
            index=0,
            line_no=line_no,
            logdir=_resolve_logdir(raw_logdir, path.parent),
            routes=routes,
            mode=mode,
            warmup=warmup,
            raw_logdir=raw_logdir,
        ))

    if not jobs:
        raise SystemExit(f"no jobs found in {path}")
    return jobs


def _dedupe_jobs(jobs: list[Job]) -> list[Job]:
    seen: set[tuple[str, tuple[str, ...], str, int | None]] = set()
    unique: list[Job] = []
    for job in jobs:
        key = (str(job.logdir), job.routes, job.mode, job.warmup)
        if key in seen:
            print(f"[queue] skipping duplicate job at line {job.line_no}: {job.raw_logdir} "
                  f"routes={';'.join(job.routes)} mode={job.mode}")
            continue
        seen.add(key)
        unique.append(job)
    return unique


def validate_jobs(jobs: list[Job], script_path: Path, python_exe: str) -> list[str]:
    errors: list[str] = []
    if not script_path.is_file():
        errors.append(f"test script not found: {script_path}")
    if shutil.which(python_exe) is None and not Path(python_exe).is_file():
        errors.append(f"python interpreter not found: {python_exe}")

    for job in jobs:
        if not job.logdir.is_dir():
            errors.append(f"line {job.line_no}: logdir not found: {job.logdir}")
            continue
        for name in ("config.py", "best_model.pth"):
            if not (job.logdir / name).is_file():
                errors.append(f"line {job.line_no}: {job.logdir} is missing {name}")
    return errors


def build_command(job: Job, script_path: Path, python_exe: str) -> list[str]:
    cmd = [
        python_exe,
        str(script_path),
        "--logdir", str(job.logdir),
    ]
    for route in job.routes:
        cmd += ["--route", route]
    cmd += ["--mode", job.mode]
    if job.warmup is not None:
        cmd += ["--warmup", str(job.warmup)]
    return cmd


def _write_stdout(chunk: bytes) -> None:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(chunk)
        stream.flush()
    else:
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()


def run_job(cmd: list[str], log_path: Path, timeout: float | None) -> tuple[int, float, bool]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    try:
        assert proc.stdout is not None
        with log_path.open("wb") as log_file:
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline = start + timeout if timeout is not None else None
            try:
                while True:
                    if deadline is None:
                        events = selector.select()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            proc.kill()
                            break
                        events = selector.select(timeout=remaining)
                        if not events:
                            timed_out = True
                            proc.kill()
                            break

                    eof = False
                    for key, _ in events:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            eof = True
                            break
                        log_file.write(chunk)
                        log_file.flush()
                        _write_stdout(chunk)
                    if eof:
                        break
            finally:
                selector.close()
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    return proc.returncode, time.monotonic() - start, timed_out


def _job_log_name(job: Job) -> str:
    raw = f"job_{job.index:03d}_{job.logdir.name}_{'_'.join(job.routes)}_{job.mode}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw) + ".log"


def write_summary(path: Path, results: list[dict[str, object]]) -> None:
    fieldnames = [
        "index", "line_no", "logdir", "route", "mode", "warmup",
        "status", "exit_code", "duration_s", "started_at", "finished_at", "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple jetson deployment benchmark jobs sequentially from a queue file."
    )
    parser.add_argument("--queue", type=Path, required=True, help="queue file with logdir, route, mode [, warmup]; separate multiple routes with ; | + or quoted whitespace")
    parser.add_argument("--run-dir", type=Path, default=None, help="directory for per-job logs and summary.csv")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for jobs (default: current interpreter)")
    parser.add_argument("--script", type=Path, default=REPO_ROOT / "jetson_deployment_test.py",
                        help="deployment test script to run (default: jetson_deployment_test.py)")

    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S,
                        help=f"seconds to wait between jobs (default: {DEFAULT_DELAY_S:g})")
    parser.add_argument("--dry-run", action="store_true", help="validate the queue and print commands without running")
    parser.add_argument("--stop-on-error", action="store_true", help="stop the queue after the first failed job")
    parser.add_argument("--timeout", type=float, default=None, help="optional per-job timeout in seconds")
    parser.add_argument("--dedupe", action="store_true", help="skip duplicate (logdir, route, mode, warmup) jobs")
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be >= 0")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")

    queue_path = args.queue.expanduser().resolve()
    if not queue_path.is_file():
        parser.error(f"queue file not found: {queue_path}")

    script_path = args.script.expanduser().resolve()
    if os.sep in args.python or args.python.startswith("."):
        python_exe = str(Path(args.python).expanduser().resolve())
    else:
        python_exe = args.python

    jobs = parse_queue(queue_path)
    if args.dedupe:
        jobs = _dedupe_jobs(jobs)
    if not jobs:
        raise SystemExit("no jobs left after deduplication")
    for index, job in enumerate(jobs, 1):
        job.index = index

    errors = validate_jobs(jobs, script_path, python_exe)
    if errors:
        print("[queue] invalid queue:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        run_dir = REPO_ROOT / "queue_logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"[queue] queue file : {queue_path}")
    print(f"[queue] jobs       : {len(jobs)}")
    print(f"[queue] script     : {script_path}")
    print(f"[queue] python     : {python_exe}")
    print(f"[queue] delay      : {args.delay:g}s between jobs")
    if args.dry_run:
        print("[queue] dry run, commands:")
        for job in jobs:
            print(f"  [{job.index}] {shlex.join(build_command(job, script_path, python_exe))}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[queue] run dir    : {run_dir}")

    results: list[dict[str, object]] = []
    interrupted = False
    try:
        for position, job in enumerate(jobs):
            if position > 0 and args.delay > 0:
                print(f"\n[queue] waiting {args.delay:g}s before next job...")
                time.sleep(args.delay)

            cmd = build_command(job, script_path, python_exe)
            log_path = run_dir / _job_log_name(job)
            print(f"\n[queue] [{position + 1}/{len(jobs)}] {job.logdir.name} "
                  f"routes={';'.join(job.routes)} mode={job.mode} warmup={job.warmup if job.warmup is not None else 'default'}")
            print(f"[queue] command: {shlex.join(cmd)}")
            print(f"[queue] log    : {log_path}")

            started_at = datetime.now()
            run_start = time.monotonic()
            try:
                returncode, duration, timed_out = run_job(cmd, log_path, args.timeout)
            except KeyboardInterrupt:
                results.append({
                    "index": job.index,
                    "line_no": job.line_no,
                    "logdir": str(job.logdir),
                    "route": ";".join(job.routes),
                    "mode": job.mode,
                    "warmup": job.warmup if job.warmup is not None else "",
                    "status": "interrupted",
                    "exit_code": "",
                    "duration_s": f"{time.monotonic() - run_start:.1f}",
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "log_path": str(log_path),
                })
                raise

            finished_at = datetime.now()
            if timed_out:
                status = "timeout"
            elif returncode == 0:
                status = "ok"
            else:
                status = "failed"

            results.append({
                "index": job.index,
                "line_no": job.line_no,
                "logdir": str(job.logdir),
                "route": ";".join(job.routes),
                "mode": job.mode,
                "warmup": job.warmup if job.warmup is not None else "",
                "status": status,
                "exit_code": returncode,
                "duration_s": f"{duration:.1f}",
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "log_path": str(log_path),
            })
            print(f"[queue] [{position + 1}/{len(jobs)}] {status.upper()} in {duration:.1f}s")

            if status != "ok" and args.stop_on_error:
                print("[queue] stopping on first failure (--stop-on-error)")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[queue] interrupted by user")

    summary_path = run_dir / "summary.csv"
    write_summary(summary_path, results)
    print(f"\n[queue] summary   : {summary_path}")

    print("[queue] results:")
    for result in results:
        print(f"  [{result['index']}] {str(result['status']).upper():10s} "
              f"{result['route']} / {result['mode']} ({result['duration_s']}s) "
              f"{Path(str(result['logdir'])).name}")

    if interrupted:
        return 130
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
