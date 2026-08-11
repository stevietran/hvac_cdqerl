#!/usr/bin/env python3
"""Torch-free GA-only demo run of the training loop -- produces saved
data on a machine without torch.

python run_qd_context_ga_demo.py                       # defaults, ~2 min
python run_qd_context_ga_demo.py --generations 100 --n-beh 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
paths.ensure_dirs()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--n-init", type=int, default=72)
    ap.add_argument("--g-ga", type=int, default=6)
    ap.add_argument("--n-beh", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--n-probe", type=int, default=800)
    ap.add_argument("--bd-calibration-genomes", type=int, default=24)
    ap.add_argument("--checkpoint-every-gen", type=int, default=20)
    ap.add_argument("--log-every-gen", type=int, default=1)
    ap.add_argument("--archive-snapshot-gens", default=None,
                    help="comma-separated generations to snapshot into "
                         "context_archive_snapshot.csv (default: every "
                         "--checkpoint-every-gen, plus 1 and the last)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1,
                    help="CPU worker processes for the rollout batch, "
                         "notes.md §8.6.4e / efficiency-plan #3 (1 = fully "
                         "serial, unchanged default)")
    args = ap.parse_args(argv)

    from hvac_qderl.scenarios import get_annual_context, get_context_archive, get_warmstart_bank
    from hvac_qderl.learners.contextual_ga_demo import run_contextual_ga_demo
    from hvac_qderl.experiments.logging_utils import RunPaths, MetricsLogger, make_run_id

    ctx = get_annual_context()
    arc = get_context_archive()
    wb = get_warmstart_bank()

    if args.archive_snapshot_gens:
        snaps = tuple(int(g) for g in args.archive_snapshot_gens.split(",") if g)
    else:
        step = max(1, args.checkpoint_every_gen)
        snaps = tuple(sorted(set(
            [1] + list(range(step, args.generations + 1, step)) + [args.generations])))

    run_paths = RunPaths(run_id=make_run_id("QD-ERL-Context-GA-Demo", args.seed))
    logger = MetricsLogger(run_paths, learner="QD-ERL-Context-GA-Demo", seed=args.seed)
    print(f"run dir: {run_paths.root}")
    print(f"archive snapshot generations: {snaps}")

    archive, hist = run_contextual_ga_demo(
        ctx.cfg, ctx.annual, ctx.load, arc, wb, hidden=args.hidden, n_beh=args.n_beh,
        n_init=args.n_init, g_ga=args.g_ga, generations=args.generations,
        n_probe=args.n_probe, bd_calibration_genomes=args.bd_calibration_genomes,
        seed=args.seed, verbose=True, logger=logger, run_paths=run_paths,
        checkpoint_every_gen=args.checkpoint_every_gen,
        log_every_gen=args.log_every_gen, archive_snapshot_gens=snaps,
        n_workers=args.n_workers)

    st = archive.stats
    print(f"\ndone: {st.num_elites}/{st.total_cells} elites "
          f"({st.coverage * 100:.1f}% coverage), QD-score {st.qd_score:.1f}")
    print(f"saved to: {run_paths.root}")
    print(f"figures:  python figures.py --only F19 F20 --run-dir {run_paths.root}")
    return run_paths.root


if __name__ == "__main__":
    main()
