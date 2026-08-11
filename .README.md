# HVAC QD-ERL — Quality‑Diversity Evolutionary Reinforcement Learning for HVAC Control

## Description

This project implements a Quality‑Diversity + Evolutionary Reinforcement Learning (QD‑ERL) approach for building HVAC control. It combines evolutionary search with reinforcement learning to discover a diverse set of robust control policies that balance energy consumption and occupant comfort. The codebase is implemented in Python and organizes experiments, learners, and dispatch routines for simulation-based evaluation.

## Visuals

Figure: algorithm flowchart from the repository docs.

![Algorithm flowchart](<docs/CQD-ERL_Flowchart-Algorithm%20QD-ERL.png>)

(See: [docs/CQD-ERL_Flowchart-Algorithm QD-ERL.png](<docs/CQD-ERL_Flowchart-Algorithm%20QD-ERL.png>))

## Prerequisites

- Python 3.8+ or 3.9+ (match `environment.yml`)
- Conda (recommended) or a compatible Python environment
- Git
- Compiler toolchain for any native dependencies (Windows: Visual C++ Build Tools)
- Enough CPU/RAM for simulations; GPU optional for neural network training

## Installation

Clone the repository and create the environment from `environment.yml`:

```bash
git clone https://github.com/yourusername/hvac_cdqerl.git
cd hvac_cdqerl
conda env create -f environment.yml -n hvac_qd_erl
conda activate hvac_qd_erl
# If you prefer pip-only:
# python -m pip install -r requirements.txt
```

If your system doesn't use Conda, recreate the packages listed in `environment.yml` with `pip` as appropriate.

## Usage

Quick examples to run experiments and dispatch scripts:

### Prepare data — Weather -> annual cooling load

```Shell
python prepare_data.py load
```

### Prepare BD space — probe-BD calibration + warm the disk caches

Context archive and warm-start bank don't need a separate command: `scenarios.get_context_archive()` and  `get_warmstart_bank()` build and cache them automatically the first time any script below touches them.

The warm-start bank's full-year G36 rollout is disk-cached under `hvac_qderl/_cache/`, so it only costs once across every subsequent run, including parallel ones.

**`iso_sigma` calibration** measures the real mutation-sigma anchor for the product archive's behaviour tessellation

Writes: `outputs/calibration/iso_sigma_calibration_<genome>.csv`

```Shell
# numpy genome (contextual_ga_demo.py) — works anywhere, no torch needed
python -m hvac_qderl.learners.calibrate_contextual_mutation_sigma --genome numpy

# torch genome (qd_erl_contextual.py) — needs torch
python -m hvac_qderl.learners.calibrate_contextual_mutation_sigma --genome torch
```

### Run CQD-ERL training

Trains the full arm: product archive`[18 context cells][N_j]`, `Iso+LineDD` + `SAC`-style policy-gradient operator, shared critics.

Writes: `runs/QD-ERL-Context_s<seed>_<timestamp>/`  with `train_log.csv`, `rollout_log.csv`, `context_archive_snapshot.csv`, `checkpoints/`, `manifest.json`.

```bash
python run_qd_context.py --probe 10                          # measure scaling first
python run_qd_context.py --seeds 0 1 2         # CPU-pooled rollouts
python run_qd_context.py --resume --total-env-steps 4_000_000
python run_qd_context.py --dry-run                            # print the plan, exit
python run_qd_context.py --status                             # runs in progress
python run_qd_context.py --describe runs/QD-ERL-Context_s0_.../
```

### Run SAC

Run the SAC training: trained on the same 18 context-cell days as the CQD-ERL arm for budget-parity comparison. Required `torch`

Writes: `runs/SAC_s<seed>_<time_stamp>` (same layout as the CQD-ERL arm)

```bash
python run_sac_parallel.py --probe 10                         # measure scaling first
python run_sac_parallel.py --seeds 0 1 2 --total-env-steps 2_000_000
python run_sac_parallel.py --seeds 0 1 2 --resume
python run_sac_parallel.py --dry-run
python run_sac_parallel.py --status
python run_sac_parallel.py --describe runs/SAC_s0_.../
```

### Dispatch

Run the annual dispatch evaluation: one continuous 8,760 h pass per controller (G36 / SAC / QD-ERL-Context)

Writes: `outputs/base/*.csv` (KPIs) + `outputs/traces/` (per-step dispatch traces, unless `--no-traces`)

```bash
python run_annual_dispatch.py                                 # G36 (if missing) + QD + SAC, seed 0
python run_annual_dispatch.py --arms qd                       # QD-ERL-Context only
python run_annual_dispatch.py --arms sac --seed 1
```

## Project structure (high level)

- `hvac_qderl/` — core library: learners, dispatch, environments, training runners
- `data/` — input datasets and representative days
- `docs/` — figures and documentation (includes algorithm flowchart used above)

## License

This repository is provided under the MIT License
