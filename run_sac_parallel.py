#!/usr/bin/env python3
"""Train the SAC arm -- self-contained training entry point.

    python run_sac_parallel.py --probe 10 --seeds 0 1   # measure scaling first
    python run_sac_parallel.py                          # seeds 0-4, 2M steps
    python run_sac_parallel.py --seeds 1 2 3 4 --resume
    python run_sac_parallel.py --describe runs/SAC_s0_.../
    python run_sac_parallel.py --dry-run
    python run_sac_parallel.py --seeds 0 1 2 --total-env-steps 1_000_000

TRAINING EPISODE (--sac-episode, default "context_set")
------------------------------------------------------
SAC now trains on THE SAME 18 DAYS as `run_qd_context.py`: all 18 context
cells resolved to their nearest real calendar day, concatenated into ONE
18 x 24 h = 432 h episode (1,296 control steps at the 20-min cadence), with
S_minpv second-last and S_peak LAST so the reward's episode-wide peak ratchet
does not freeze over the 16 ordinary days.

What is now IDENTICAL across the two arms: the environment (`HVACPlantEnv` on
`get_annual_context().cfg`), the observation (`common.augment_obs`, 66-D), the
action (`common.action_from_vector`, 3-D), the reward, and the day set.

What  DIFFERS, by design:

  episode length   SAC 18 days per episode       |  QD 1 day per rollout
  reset state      isothermal T_z = T_m = T_set  |  warm-start bank draw
  objective        raw episode return Sigma r    |  F_tilde vs the F_G36 day baseline
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
paths.ensure_dirs()

from hvac_qderl.experiments.launcher import main

DEFAULTS = {'--learner': 'sac'}

if __name__ == "__main__":
    argv = sys.argv[1:]
    for flag, value in DEFAULTS.items():
        if flag not in argv:
            argv = [flag, value] + argv
    # `entry` is this file: children are re-exec'd from here in --worker mode,
    sys.exit(main(argv, entry=os.path.abspath(__file__)))
