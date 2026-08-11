#!/usr/bin/env python3
"""Concurrent multi-seed training launcher, shared by the three run_*.py CLIs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # repo root, not the package dir
RUNS = os.path.join(HERE, "runs")
LOGS = os.path.join(HERE, "logs")

# torch-free import (learners/common.py -> learners/policy.py only), safe to
# do in the parent process which never itself needs torch.
from ..learners.common import ACT_DIM as EXPECTED_ACT_DIM

NAME = {"sac": "SAC", "qdcontext": "QD-ERL-Context"}

# Sequential single-process baselines, MEASURED on this machine class.
BASELINES = {"sac": 28.7}

STEP_RE = re.compile(r"steps\s+([\d,]+)\s*\|\s*eval\s+(-?[\d.]+)\s*\|\s*([\d.]+)s")
GEN_RE = re.compile(r"gen\s+(\d+)\D+?([\d,]+)\s*steps", re.I)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def human_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024 ** 3:.1f} GB"


def replay_bytes_per_proc(capacity=1_000_000, feat_dim=60, act_dim=3) -> int:
    """float32 arrays in learners/replay.py: obs, next_obs, act, rew, done, h, h_next."""
    # 2*feat + act + rew + done. The 2*belief_dim hidden-state columns were
    # dropped with the GRU encoder: -256 MB/process at capacity.
    per_row = 2 * feat_dim + act_dim + 1 + 1
    return capacity * per_row * 4


def probe_env(py: str) -> dict:
    """Ask a child process about torch/CUDA/action space. Returns {} on failure."""
    code = r"""
import json, sys
out = {}
try:
    from hvac_qderl.learners.common import ACT_DIM
    out["act_dim"] = ACT_DIM
except Exception as e:
    out["import_error"] = f"{type(e).__name__}: {e}"
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.cuda.is_available()
    if out["cuda"]:
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_free"], out["gpu_total"] = free, total
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""
    try:
        r = subprocess.run([py, "-c", code], cwd=HERE, capture_output=True,
                           text=True, timeout=180)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def active_runs(learner: str, max_age_s: float = 600.0) -> dict:
    """Seeds whose manifest.json was touched recently => probably still training."""
    out = {}
    if not os.path.isdir(RUNS):
        return out
    prefix = NAME[learner] + "_s"
    now = time.time()
    for d in os.listdir(RUNS):
        if not d.startswith(prefix):
            continue
        man = os.path.join(RUNS, d, "manifest.json")
        if not os.path.isfile(man):
            continue
        age = now - os.path.getmtime(man)
        if age > max_age_s:
            continue
        try:
            m = json.load(open(man))
        except Exception:
            continue
        seed = int(m.get("seed", d.split("_s")[1].split("_")[0]))
        prev = out.get(seed)
        if prev is None or m.get("env_steps", 0) > prev["env_steps"]:
            out[seed] = {"dir": d, "env_steps": m.get("env_steps", 0),
                         "age_s": age, "device": m.get("device")}
    return out


def latest_run_for(learner: str, seed: int) -> str:
    """Newest run dir for this learner+seed that has a resumable checkpoint."""
    if not os.path.isdir(RUNS):
        return None
    pref = f"{NAME[learner]}_s{seed}_"
    cands = [d for d in os.listdir(RUNS) if d.startswith(pref)
             and os.path.isfile(os.path.join(RUNS, d, "latest.json"))]
    if not cands:
        return None
    newest = max(cands, key=lambda d: os.path.getmtime(
        os.path.join(RUNS, d, "latest.json")))
    return os.path.join(RUNS, newest)


def preflight(args, py: str) -> dict:
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    info = probe_env(py)

    if "import_error" in info:
        print(f"  X cannot import hvac_qderl: {info['import_error']}")
        sys.exit(1)

    if info.get("act_dim") != EXPECTED_ACT_DIM:
        print(f"  X expected ACT_DIM={EXPECTED_ACT_DIM}, found {info.get('act_dim')}")
        sys.exit(1)
    print(f"  ACT_DIM        : {info['act_dim']}")

    needs_torch = args.learner in ("sac", "qdcontext")
    if needs_torch:
        if "torch" not in info:
            print(f"  X torch required for '{args.learner}': "
                  f"{info.get('torch_error', 'not installed')}")
            sys.exit(1)
        print(f"  torch          : {info['torch']}  CUDA={info.get('cuda')}")
        if info.get("cuda"):
            print(f"  GPU            : {info.get('gpu')}  "
                  f"free {human_gb(info['gpu_free'])} / "
                  f"{human_gb(info['gpu_total'])}")
        else:
            print("  ! CUDA unavailable - concurrency still helps, but each "
                  "process now competes for CPU.")

    # --- the two resources concurrency actually contends for --------------- #
    n = len(args.seeds) if args.max_parallel <= 0 else min(args.max_parallel,
                                                           len(args.seeds))
    rb = replay_bytes_per_proc()
    # torch + CUDA host context is roughly 1.2 GB per process in practice
    est = n * (rb + int(1.2 * 1024 ** 3))
    print(f"  replay buffer  : {human_gb(rb)}/proc  "
          f"-> ~{human_gb(est)} peak RSS for {n} concurrent")
    try:
        import psutil  # optional
        avail = psutil.virtual_memory().available
        print(f"  system RAM free: {human_gb(avail)}")
        if est > 0.8 * avail:
            print(f"  ! estimated use exceeds 80 % of free RAM. "
                  f"Lower --max-parallel (currently {n}).")
    except ImportError:
        print("  system RAM free: unknown (pip install psutil to check)")

    if needs_torch and info.get("cuda"):
        # each CUDA context is ~300-500 MB of device memory
        gpu_est = n * 500 * 1024 ** 2
        if gpu_est > 0.8 * info["gpu_free"]:
            print(f"  ! ~{human_gb(gpu_est)} of GPU context for {n} procs vs "
                  f"{human_gb(info['gpu_free'])} free. Lower --max-parallel.")

    cores = os.cpu_count() or 1
    total_threads = n * args.threads_per_proc
    print(f"  CPU            : {cores} cores | {n} procs x "
          f"{args.threads_per_proc} threads = {total_threads}")
    if total_threads > cores:
        print(f"  ! {total_threads} threads on {cores} cores will contend. "
              f"Reduce --threads-per-proc or --max-parallel.")

    # --- CPU HEADROOM ---------------------------------------------------- #
    # This is the binding constraint, and it is counter-intuitive: a
    # latency-bound CUDA workload is CPU-hungry. SAC issues thousands of tiny
    # kernel launches per gradient step (twin-Q + targets + policy + alpha,
    # then Adam over many small tensors); every launch is CPU time in the
    # driver. A free GPU does not help if no core is free to feed it.
    free_cores = None
    try:
        import psutil
        util = psutil.cpu_percent(interval=2.0)
        free_cores = cores * (1.0 - util / 100.0)
        print(f"  CPU load       : {util:.0f} % busy "
              f"-> ~{free_cores:.1f} of {cores} cores free")
        rec = max(1, int(free_cores))
        if free_cores < n:
            print(f"  ! Only ~{free_cores:.1f} cores are free but you asked for "
                  f"{n} concurrent processes.")
            print(f"    Each SAC process needs roughly one full core to issue "
                  f"CUDA launches, so")
            print(f"    the extra processes will not speed anything up - they "
                  f"will slow each other down.")
            print(f"    Recommended: --max-parallel {rec}")
            print(f"    Better: find and pause whatever is using the CPU; this "
                  f"workload is CPU-bound,")
            print(f"    so freeing cores speeds up even a SINGLE seed.")
        if util > 60 and BASELINES.get(args.learner):
            print(f"  ! The {BASELINES[args.learner]} steps/s baseline was "
                  f"measured under whatever load existed then.")
            print(f"    On a busy machine the true idle-machine rate is likely "
                  f"HIGHER - run --probe to find out.")
    except ImportError:
        print("  CPU load       : unknown (pip install psutil - strongly "
              "recommended before scaling out)")

    live = active_runs(args.learner)
    if live:
        print(f"  ! {len(live)} {NAME[args.learner]} run(s) look ACTIVE "
              f"(manifest touched < 10 min ago):")
        for s, v in sorted(live.items()):
            print(f"      seed {s}: {v['env_steps']:,} steps, "
                  f"updated {v['age_s']:.0f}s ago  ({v['dir']})")
        clash = sorted(set(live) & set(args.seeds))
        if clash and not args.allow_duplicate:
            print(f"  X seeds {clash} are already training. Two processes on the "
                  f"same seed waste a slot and race on the run dir.")
            print(f"    Either wait, or drop them:  --seeds "
                  f"{' '.join(str(s) for s in args.seeds if s not in clash)}")
            print("    Override with --allow-duplicate if you know better.")
            sys.exit(1)
    return info


# --------------------------------------------------------------------------- #
# job management
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    seed: int
    cmd: list
    log: str
    proc: subprocess.Popen = None
    t0: float = 0.0
    t_end: float = 0.0
    rc: int = None
    last: tuple = field(default=None)      # (steps, eval, run_seconds)
    run_dir: str = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def poll_log(self):
        """Best available progress for this job.

        Two sources, because neither alone is sufficient:
          * the log's progress line - fine-grained (every 5 k steps for SAC) but
            its format differs per learner and may not exist yet;
          * the run's manifest.json - authoritative and format-independent, but
            only rewritten at checkpoints (every 50 k steps for SAC).
        Take whichever reports more progress.
        """
        head = tail = ""
        try:
            size = os.path.getsize(self.log)
            with open(self.log, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4096)
                if size > 8192:
                    fh.seek(size - 8192)
                tail = head + fh.read()
        except OSError:
            return

        # the run dir is announced on line 1 by every learner
        if self.run_dir is None:
            m = re.search(r"run dir:\s*(.+)", head)
            if m:
                self.run_dir = m.group(1).strip()

        steps = ev = None
        m = None
        for m in STEP_RE.finditer(tail):
            pass
        if m:
            steps, ev = int(m.group(1).replace(",", "")), float(m.group(2))
        else:
            for m in GEN_RE.finditer(tail):
                pass
            if m:
                steps, ev = int(m.group(2).replace(",", "")), float("nan")

        man_steps = self._manifest_steps()
        if man_steps is not None and (steps is None or man_steps > steps):
            steps, ev = man_steps, (ev if ev is not None else float("nan"))

        if steps is not None:
            self.last = (steps, ev if ev is not None else float("nan"),
                         time.time() - self.t0)

    def _manifest_steps(self):
        if not self.run_dir:
            return None
        for fname in ("manifest.json", "latest.json"):
            p = os.path.join(self.run_dir, fname)
            try:
                with open(p) as fh:
                    return int(json.load(fh).get("env_steps", 0)) or None
            except Exception:
                continue
        return None


def build_cmd(py: str, args, seed: int) -> list:
    cmd = [py, "-u", os.path.abspath(args.entry), "--worker",
           "--learner", args.learner,
           "--total-env-steps", str(args.total_env_steps),
           "--seed", str(seed)]
    if args.learner == "qdcontext":
        cmd += ["--checkpoint-every-gen", "25", "--log-every-gen", "10",
                "--archive-snapshot-gens", "1,50,100,200,300",
                "--n-beh", str(args.n_beh),
                "--n-workers", str(args.n_workers)]
    if args.learner == "sac":
        cmd += ["--log-every", str(args.log_every),
                "--checkpoint-every", str(args.checkpoint_every),
                "--sac-episode", str(args.sac_episode)]
    if args.cpu:
        cmd += ["--cpu"]
    if args.resume:
        run = latest_run_for(args.learner, seed)
        if run:
            cmd += ["--resume-from", run]
    return cmd


def child_env(args) -> dict:
    env = os.environ.copy()
    t = str(args.threads_per_proc)
    # Read once, at import. Setting them after torch/numpy load has no effect,
    # which is exactly why this is done in the parent before spawning.
    env.update({"OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t,
                "OPENBLAS_NUM_THREADS": t, "NUMEXPR_NUM_THREADS": t,
                "VECLIB_MAXIMUM_THREADS": t,
                "PYTHONUNBUFFERED": "1",           # keeps the live table honest
                "HVAC_N_REP_DAYS": str(args.n_rep_days), # sac only; ignored by qdcontext
                "HVAC_SAC_EPISODE": str(args.sac_episode)})
    return env


def launch(job: Job, env: dict) -> None:
    fh = open(job.log, "w", encoding="utf-8", buffering=1)
    job._fh = fh
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    job.proc = subprocess.Popen(job.cmd, cwd=HERE, env=env, stdout=fh,
                                stderr=subprocess.STDOUT,
                                creationflags=creationflags)
    job.t0 = time.time()
    if PRIORITY != "normal":
        set_priority(job.proc.pid, PRIORITY)


def set_priority(pid: int, level: str) -> None:
    """Raise or lower scheduling priority of a child.

    Useful when the box is already loaded: 'high' lets training win CPU against
    background work; 'low' does the opposite, letting training soak up only idle
    cycles so it does not make the machine unusable.
    """
    try:
        import psutil
    except ImportError:
        print(f"  ! --priority {level} needs psutil; leaving priority normal.")
        return
    try:
        p = psutil.Process(pid)
        if os.name == "nt":
            p.nice({"high": psutil.HIGH_PRIORITY_CLASS,
                    "low": psutil.IDLE_PRIORITY_CLASS}[level])
        else:
            p.nice({"high": -5, "low": 10}[level])
    except Exception as e:
        print(f"  ! could not set priority on pid {pid}: "
              f"{type(e).__name__}: {e}")
        if level == "high" and os.name != "nt":
            print("    (raising priority on Linux needs root; "
                  "try --priority low on the competing job instead)")


def stop_all(jobs, sig="terminate"):
    for j in jobs:
        if j.running:
            try:
                j.proc.terminate() if sig == "terminate" else j.proc.kill()
            except Exception:
                pass
    t0 = time.time()
    while any(j.running for j in jobs) and time.time() - t0 < 20:
        time.sleep(0.5)
    for j in jobs:
        if j.running:
            try:
                j.proc.kill()
            except Exception:
                pass


def status_table(jobs, t_start) -> str:
    el = time.time() - t_start
    out = [f"\n  elapsed {el / 3600:6.2f} h   "
           f"{sum(j.running for j in jobs)} running / {len(jobs)} total",
           f"  {'seed':>5} {'state':>9} {'steps':>12} {'steps/s':>9} "
           f"{'eval':>12} {'ETA':>8}"]
    agg = 0.0
    for j in sorted(jobs, key=lambda x: x.seed):
        if j.proc is None:
            out.append(f"  {j.seed:>5} {'queued':>9} {'-':>12} {'-':>9} "
                       f"{'-':>12} {'-':>8}")
            continue
        j.poll_log()
        run_s = (j.t_end or time.time()) - j.t0
        steps = j.last[0] if j.last else 0
        rate = steps / run_s if run_s > 0 and steps else 0.0
        if j.running:
            agg += rate
        state = "running" if j.running else (
            "done" if j.rc == 0 else f"exit {j.rc}")
        ev = f"{j.last[1]:,.0f}" if j.last and j.last[1] == j.last[1] else "-"
        eta = "-"
        if j.running and rate > 0:
            eta = f"{(TOTAL - steps) / rate / 3600:.1f} h"
        out.append(f"  {j.seed:>5} {state:>9} {steps:>12,} {rate:>9.2f} "
                   f"{ev:>12} {eta:>8}")
    base = BASELINES.get(LEARNER)
    ref = f" (sequential baseline {base})" if base else ""
    out.append(f"  {'':>5} {'aggregate':>9} {'':>12} {agg:>9.2f} steps/s{ref}")
    return "\n".join(out)


TOTAL = 2_000_000     
LEARNER = "sac"        
PRIORITY = "normal"     


# --------------------------------------------------------------------------- #
def cmd_status(args):
    live = active_runs(args.learner, max_age_s=args.stale_after)
    if not live:
        print(f"No {NAME[args.learner]} runs updated in the last "
              f"{args.stale_after:.0f}s.")
        return 0
    print(f"{'seed':>5} {'steps':>12} {'updated':>10}  run dir")
    for s, v in sorted(live.items()):
        print(f"{s:>5} {v['env_steps']:>12,} {v['age_s']:>9.0f}s  {v['dir']}")
    return 0


def main(argv=None, entry: str | None = None) -> int:
    global TOTAL
    p = argparse.ArgumentParser(
        allow_abbrev=False,  
        description="Run training seeds concurrently.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=None,
                   help="single-seed shorthand for --seeds; also the seed a "
                        "--worker child trains")
    p.add_argument("--learner", default="qdcontext",
                   choices=["sac", "qdcontext"])
    p.add_argument("--total-env-steps", type=int, default=2_000_000)
    p.add_argument("--max-parallel", type=int, default=0,
                   help="0 = all seeds at once")
    p.add_argument("--threads-per-proc", type=int, default=1,
                   help="OMP/MKL threads per child. 1 is right when the GPU "
                        "does the work.")
    p.add_argument("--n-beh", type=int, default=12,
                   help="qdcontext only: behaviour cells per context cell "
                        "(archive = 18 x n-beh, notes.md §8.5)")
    p.add_argument("--n-workers", type=int, default=1,
                   help="qdcontext only: CPU worker processes for the "
                        "rollout batch, notes.md §8.6.4e / efficiency-plan "
                        "#3 (1 = fully serial, unchanged default)")
    p.add_argument("--n-rep-days", type=int, default=4)
    p.add_argument("--sac-episode", default="context_set",
                   choices=["context_set", "representative"],
                   help="sac only: training episode design. 'context_set' "
                        "(default) runs all 18 context-cell days in one 432 h / "
                        "1,296-step episode -- the same day set run_qd_context.py "
                        "trains on. 'representative' is the legacy n_rep_days + "
                        "min-PV + peak-load 6-day / 432-step episode. NOTE the "
                        "two are not step-for-step budget-comparable: 2M steps "
                        "is 1,543 context-set episodes vs 4,630 representative "
                        "ones. --n-rep-days has no effect under 'context_set'.")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--poll", type=float, default=60.0,
                   help="seconds between status updates")
    p.add_argument("--probe", type=float, default=0.0, metavar="MIN",
                   help="run for MIN minutes, report measured scaling, stop")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true",
                   help="report on in-progress runs and exit")
    p.add_argument("--stale-after", type=float, default=600.0)
    p.add_argument("--allow-duplicate", action="store_true",
                   help="permit launching a seed that already looks active")
    p.add_argument("--resume", action="store_true",
                   help="continue each seed from its newest checkpoint if one "
                        "exists; seeds with no prior run start fresh")
    p.add_argument("--priority", default="normal",
                   choices=["normal", "high", "low"],
                   help="child scheduling priority. 'high' wins CPU against "
                        "background load; 'low' keeps the machine responsive.")
    p.add_argument("--python", default=sys.executable)
    # --- worker-only knobs (consumed by train_worker.train in the child) --- #
    p.add_argument("--generations", type=int, default=100_000)
    p.add_argument("--checkpoint-every-gen", type=int, default=25)
    p.add_argument("--log-every-gen", type=int, default=10)
    p.add_argument("--archive-snapshot-gens", default="1,50,100,200,300")
    p.add_argument("--resume-from", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true",
                   help=argparse.SUPPRESS)      # internal: train ONE seed here
    p.add_argument("--describe", metavar="RUN_DIR", default=None,
                   help="print a saved agent's provenance and exit")
    args = p.parse_args(argv)
    # the file to re-exec for each child; each CLI passes its own __file__ so
    # there is no shared orchestrator script to keep in sync
    args.entry = entry or os.path.abspath(sys.argv[0])

    if args.describe:
        from .train_worker import describe
        args.run = args.describe
        describe(args)
        return 0
    if args.worker:
        # child process: do the actual training for this one seed
        if args.seed is None:
            args.seed = args.seeds[0]
        from .train_worker import train
        train(args)
        return 0

    if args.seed is not None:
        args.seeds = [args.seed]

    if args.status:
        return cmd_status(args)

    global LEARNER, PRIORITY
    TOTAL = args.total_env_steps
    LEARNER, PRIORITY = args.learner, args.priority
    os.makedirs(LOGS, exist_ok=True)
    py = args.python or "python"

    # what a *real* run would cost, kept for the probe's extrapolation
    args.full_steps = args.total_env_steps
    if args.probe:
        # a probe must not stop on step count; it is bounded by wall-clock only
        args.total_env_steps = 10 ** 9
        TOTAL = args.total_env_steps

    preflight(args, py)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    n_par = len(args.seeds) if args.max_parallel <= 0 else args.max_parallel
    jobs = [Job(seed=s, cmd=build_cmd(py, args, s),
                log=os.path.join(LOGS, f"par_{args.learner}_s{s}_{stamp}.log"))
            for s in args.seeds]

    print("\n" + "=" * 78)
    print("PLAN")
    print("=" * 78)
    print(f"  learner        : {args.learner}")
    print(f"  seeds          : {args.seeds}")
    print(f"  concurrency    : {n_par} at a time")
    print(f"  budget         : {args.total_env_steps:,} env steps"
          if not args.probe else
          f"  budget         : (probe: {args.probe:g} min, then stop)")
    print(f"  threads/proc   : {args.threads_per_proc}")
    print(f"  priority       : {args.priority}")
    print(f"  logs           : {LOGS}")
    print(f"  example cmd    : {' '.join(jobs[0].cmd)}")

    if args.dry_run:
        print("\nDRY RUN - nothing launched.")
        for j in jobs:
            print(f"  {' '.join(j.cmd)}")
        return 0

    env = child_env(args)
    t_start = time.time()
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, on_sigint)

    print("\n" + "=" * 78)
    print("RUNNING   (Ctrl-C stops all children cleanly)")
    print("=" * 78)

    pending = list(jobs)
    try:
        while True:
            while pending and sum(j.running for j in jobs) < n_par:
                j = pending.pop(0)
                launch(j, env)
                print(f"  launched seed {j.seed}  pid {j.proc.pid}  -> {j.log}")
                time.sleep(2.0)   # stagger: avoids a CUDA-init thundering herd

            for j in jobs:
                if j.proc is not None and j.rc is None and not j.running:
                    j.rc = j.proc.returncode
                    j.t_end = time.time()
                    tag = "OK" if j.rc == 0 else f"FAILED rc={j.rc}"
                    print(f"  seed {j.seed} {tag} after "
                          f"{(j.t_end - j.t0) / 3600:.2f} h")

            if interrupted:
                print("\n  interrupt received - stopping children ...")
                break
            if args.probe and time.time() - t_start >= args.probe * 60:
                print(f"\n  probe window ({args.probe:g} min) elapsed - stopping ...")
                break
            if not pending and not any(j.running for j in jobs):
                break

            print(status_table(jobs, t_start), flush=True)
            slept = 0.0
            while slept < args.poll and not interrupted:
                time.sleep(1.0)
                slept += 1.0
    finally:
        stop_all(jobs)
        for j in jobs:
            fh = getattr(j, "_fh", None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    # ----------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    rates = []
    for j in sorted(jobs, key=lambda x: x.seed):
        j.poll_log()
        run_s = (j.t_end or time.time()) - j.t0 if j.t0 else 0
        steps = j.last[0] if j.last else 0
        rate = steps / run_s if run_s > 0 and steps else 0.0
        if rate > 0:
            rates.append(rate)
        print(f"  seed {j.seed}: {steps:>10,} steps in {run_s / 60:7.1f} min"
              f"  = {rate:6.2f} steps/s   log: {os.path.basename(j.log)}")

    if rates:
        agg = sum(rates)
        per = agg / len(rates)
        base = BASELINES.get(args.learner)
        print(f"\n  per-process : {per:.2f} steps/s")
        print(f"  aggregate   : {agg:.2f} steps/s across {len(rates)} processes")
        if base:
            print(f"  vs baseline : {per / base * 100:.0f} % of the {base} "
                  f"steps/s single-process rate")
            print(f"  speedup     : {agg / base:.2f}x vs one seed at a time")
        else:
            print(f"  (no single-process baseline recorded for "
                  f"'{args.learner}' - speedup not computed)")
        if args.probe:
            full = args.full_steps
            hrs = full / per / 3600 if per else float("nan")
            print(f"\n  At this rate a full {full:,}-step run takes {hrs:.1f} h,")
            print(f"  and all {len(rates)} seeds finish together in ~{hrs:.1f} h.")
            if base:
                print(f"  Sequentially that would be "
                      f"{len(rates) * full / base / 3600:.0f} h.")
            if base and per < 0.6 * base:
                print("\n  ! Per-process throughput dropped >40 %: the processes "
                      "are contending.")
                print("    Lower --max-parallel, or confirm --threads-per-proc "
                      "is 1.")
    failed = [j.seed for j in jobs if j.rc not in (0, None)]
    if failed and not args.probe:
        print(f"\n  ! failed seeds: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
