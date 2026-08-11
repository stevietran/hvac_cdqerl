"""Torch-free GA-only demonstration of the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import feature_dim_for, evaluate_genome_contextual, ACT_DIM
from .policy import make_policy
from .product_archive import ProductArchive, calibrate_probe_bd_ranges


@dataclass
class GADemoHistory:
    generations: list = field(default_factory=list)
    env_steps: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    qd_score: list = field(default_factory=list)
    obj_max: list = field(default_factory=list)


def run_contextual_ga_demo(cfg, annual, load, context_archive, warmstart_bank,
                           hidden: int = 24, n_beh: int = 6, n_init: int = 40,
                           g_ga: int = 8, generations: int = 25,
                           iso_sigma: float = 0.0774, line_sigma: float = 0.2,
                           n_probe: int = 800, bd_calibration_genomes: int = 16,
                           seed: int = 0, verbose: bool = False,
                           logger=None, run_paths=None,
                           checkpoint_every_gen: int = 0, log_every_gen: int = 1,
                           archive_snapshot_gens=(), n_workers: int = 1,
                           mos_path: str | None = None,
                           price_profile: str | None = None):
    rng = np.random.default_rng(seed)
    obs_dim = feature_dim_for(cfg)
    n_par = make_policy(obs_dim, ACT_DIM, hidden).n_params

    from ..probe_bd import build_probe_grid
    grid = build_probe_grid(cfg, len(cfg.zones), n_probe=n_probe, seed=seed)
    bd_ranges = calibrate_probe_bd_ranges(cfg, n_genomes=bd_calibration_genomes,
                                          hidden=hidden, seed=seed)
    archive = ProductArchive(solution_dim=n_par, bd_ranges=bd_ranges,
                             n_context_cells=18, n_beh=n_beh, seed=seed)

    env_steps = 0
    pool = None
    if n_workers and n_workers > 1:
        from .rollout_pool import RolloutPool
        pool = RolloutPool(n_workers=n_workers, kind="numpy", mos_path=mos_path,
                           price_profile=price_profile, seed=seed, hidden=hidden,
                           n_probe=n_probe, n_context_cells=18)

    def _eval(theta, generation, source=""):
        nonlocal env_steps
        F_tilde, bd, cell, day, F_raw = evaluate_genome_contextual(
            theta, cfg, annual, load, context_archive, warmstart_bank, rng,
            hidden=hidden, probe_grid=grid)
        env_steps += 72          # 24h / 20-min control steps
        if logger is not None:
            logger.log_rollout(generation=generation, env_steps=env_steps,
                               context_cell=cell, day=day, fitness_raw=F_raw,
                               f_g36=warmstart_bank.f_g36(day),
                               fitness_tilde=F_tilde, source=source)
        return F_tilde, bd, cell

    def _eval_batch(thetas, generation, sources=None):
        nonlocal env_steps
        if not thetas:
            return []
        sources = list(sources) if sources is not None else [""] * len(thetas)
        if pool is None:
            return [_eval(th, generation, s) for th, s in zip(thetas, sources)]
        results = pool.submit_batch(thetas, generation)
        out = []
        for r, s in zip(results, sources):
            env_steps += 72
            if logger is not None:
                logger.log_rollout(generation=r["generation"], env_steps=env_steps,
                                   context_cell=r["cell"], day=r["day"],
                                   fitness_raw=r["F_raw"], f_g36=r["f_g36"],
                                   fitness_tilde=r["F_tilde"], source=s)
            out.append((r["F_tilde"], r["bd"], r["cell"]))
        return out

    try:
        init_thetas = [rng.normal(0, 0.5, n_par) for _ in range(n_init)]
        init_src = ["bootstrap"] * len(init_thetas)
        for th, (F, bd, cell) in zip(init_thetas,
                                     _eval_batch(init_thetas, 0, init_src)):
            archive.add(cell, th, F, bd)

        hist = GADemoHistory()
        for gen in range(1, generations + 1):
            elites = archive.all_solutions()
            cand, cand_src = [], []
            for _ in range(g_ga):
                if len(elites) < 2:
                    cand.append(rng.normal(0, 0.5, n_par))
                    cand_src.append("random_fallback")
                else:
                    i, j = rng.integers(0, len(elites), 2)
                    cand.append(elites[i] + iso_sigma * rng.normal(size=n_par)
                               + line_sigma * (elites[j] - elites[i]) * rng.normal())
                    cand_src.append("iso_line")
            for th, (F, bd, cell) in zip(cand, _eval_batch(cand, gen, cand_src)):
                archive.add(cell, th, F, bd)

            st = archive.stats
            hist.generations.append(gen)
            hist.env_steps.append(env_steps)
            hist.coverage.append(st.coverage)
            hist.qd_score.append(st.qd_score)
            hist.obj_max.append(st.obj_max)

            if logger is not None and (gen % log_every_gen == 0 or gen == 1):
                logger.log_train(env_steps=env_steps, generation=gen, grad_steps=0,
                                 best_return=st.obj_max, mean_return=st.obj_mean,
                                 eval_return=st.obj_max, coverage=st.coverage,
                                 qd_score=st.qd_score, archive_elites=st.num_elites,
                                 buffer_size=0, alpha=None)
            elif verbose:
                print(f"  gen {gen:3d} | steps {env_steps:6d} | "
                      f"coverage {st.coverage:.3f} | QD {st.qd_score:8.2f} | "
                      f"best F~ {st.obj_max:6.3f}")
            if logger is not None and gen in tuple(archive_snapshot_gens):
                logger.log_context_archive(archive, gen, env_steps)
            if run_paths is not None and checkpoint_every_gen and \
                    gen % checkpoint_every_gen == 0:
                _save(run_paths, logger, archive, seed, env_steps, gen,
                     hidden=hidden, n_beh=n_beh, n_workers=n_workers, is_best=False)

        if run_paths is not None:
            if logger is not None:
                logger.log_context_archive(archive, generations, env_steps)
            _save(run_paths, logger, archive, seed, env_steps, generations,
                 hidden=hidden, n_beh=n_beh, n_workers=n_workers, is_best=True)

        return archive, hist
    finally:
        if pool is not None:
            pool.shutdown()


def _save(run_paths, logger, archive, seed, env_steps, generation, hidden,
         n_beh, is_best, n_workers: int = 1):
    from ..experiments.checkpoint import save_checkpoint
    st = archive.stats
    return save_checkpoint(
        run_paths, learner="QD-ERL-Context-GA-Demo", seed=seed,
        env_steps=env_steps, generation=generation, grad_steps=0,
        wall_clock_s=logger.wall_clock if logger else 0.0,
        agent=None, archive=archive,
        config={"hidden": hidden, "n_beh": n_beh, "n_workers": n_workers},
        metrics={"best_eval_return": round(float(st.obj_max), 4),
                 "qd_score": round(float(st.qd_score), 3),
                 "coverage": round(st.coverage, 5),
                 "archive_elites": int(st.num_elites)},
        archive_kind="product", is_best=is_best)
