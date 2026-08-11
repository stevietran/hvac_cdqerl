#!/usr/bin/env python3
"""Training worker and provenance printer, shared by the three run_*.py CLIs.

`train()` runs in a CHILD process launched by `experiments.launcher` (one process
per seed, for concurrency and CPU-thread isolation), or directly via `--worker`.
"""
from __future__ import annotations

import os
import sys

from ..config import default_singapore_config
from .logging_utils import RunPaths, MetricsLogger, make_run_id


def _resume_run_root(resume: str) -> str:
    """Run directory that a --resume argument refers to.

    Accepts all three forms the docs and the launchers use:
        runs/<run>/                        -> itself
        runs/<run>/latest.json             -> its parent
        runs/<run>/checkpoints/ckpt_*.pt   -> its grandparent
    """
    resume = os.path.abspath(resume)
    if os.path.isdir(resume):
        return resume
    parent = os.path.dirname(resume)
    if os.path.basename(parent).lower() == "checkpoints":
        return os.path.dirname(parent)
    return parent


def train(a):
    learner = a.learner.lower()
    name = {"sac": "SAC", "qdcontext": "QD-ERL-Context"}[learner]

    # resuming reuses the *same* run dir so train_log.csv stays one continuous history
    if a.resume_from:
        run_root = _resume_run_root(a.resume_from)
        paths = RunPaths(run_id=os.path.basename(run_root.rstrip("/\\")),
                         root=run_root)
    else:
        paths = RunPaths(run_id=make_run_id(name, a.seed))
    logger = MetricsLogger(paths, learner=name, seed=a.seed)
    print(f"run dir: {paths.root}")

    if learner == "sac":
        from hvac_qderl.learners.sac import train_sac, SACConfig
        train_sac(total_steps=a.total_env_steps,
                  cfg=SACConfig(prefer_gpu=not a.cpu),
                  seed=a.seed, log_every=a.log_every,
                  run_paths=paths, logger=logger,
                  checkpoint_every=a.checkpoint_every,
                  resume_from=a.resume_from,
                  episode=getattr(a, "sac_episode", None))

    elif learner == "qdcontext":
        from hvac_qderl.learners.qd_erl_contextual import (run_contextual_qd_erl,
                                                            ContextualQDERLConfig)
        snaps = tuple(int(g) for g in a.archive_snapshot_gens.split(",") if g)
        run_contextual_qd_erl(generations=a.generations,
                              total_env_steps=a.total_env_steps,
                              qcfg=ContextualQDERLConfig(prefer_gpu=not a.cpu,
                                                         n_beh=a.n_beh,
                                                         n_workers=a.n_workers),
                              seed=a.seed, logger=logger, run_paths=paths,
                              checkpoint_every_gen=a.checkpoint_every_gen,
                              log_every_gen=a.log_every_gen,
                              archive_snapshot_gens=snaps,
                              resume_from=a.resume_from)

    else:
        raise ValueError(f"unknown learner {learner!r}")

    print(f"\nsaved. inspect with:  python run_comparison.py describe {paths.root}")
    return paths.root



def describe(a):
    """Print a saved agent's provenance (`--describe RUN_DIR`)."""
    from .checkpoint import describe as _describe
    print(_describe(a.run))

