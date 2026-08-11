#!/usr/bin/env python3
"""Train the contextual QD-ERL arm -- self-contained training entry point

    python run_qd_context.py --probe 10 --seeds 0 1
    python run_qd_context.py                              # seeds 0-4, 2M steps
    python run_qd_context.py --total-env-steps 1_000_000 --seeds 0 1 2
    
    python run_qd_context.py --describe runs/QD-ERL-Context_s0_.../
    python run_qd_context.py --dry-run
    python run_qd_context.py --total-env-steps 1_000_000 --seeds 0 1 2
    python run_qd_context.py --resume --total-env-steps 2_000_000

COMMON OPTIONS (all four runners share one engine: `experiments.launcher`)
---------------------------------------------------------------------------
  --seeds 0 1 2 3 4      which seeds            --seed N   single-seed shorthand
  --total-env-steps N    budget per seed (default 2,000,000; NOT comparable
                         1:1 against run_qderl.py -- see above)
  --max-parallel N       concurrent processes   --threads-per-proc N  OMP threads
  --probe MIN            run MIN minutes, report measured scaling, stop
  --resume               continue each seed from its newest checkpoint
  --priority high|low    scheduling priority against other work on the box
  --cpu / --force-cpu    keep torch off the GPU
  --dry-run              print the plan and exit
  --status               report runs currently in progress
  --describe RUN_DIR     print a saved agent's provenance and exit
  --n-beh N              behaviour cells per context cell
  --n-workers N          CPU worker processes for the rollout batch (1 = serial, default;)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
paths.ensure_dirs()

from hvac_qderl.experiments.launcher import main

DEFAULTS = {'--learner': 'qdcontext'}

if __name__ == "__main__":
    argv = sys.argv[1:]
    for flag, value in DEFAULTS.items():
        if flag not in argv:
            argv = [flag, value] + argv
    # `entry` is this file: children are re-exec'd from here in --worker mode,
    sys.exit(main(argv, entry=os.path.abspath(__file__)))
