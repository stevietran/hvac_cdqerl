"""the contextual-QD training loop.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import (feature_dim_for, augment_obs, action_from_vector,
                     sample_training_context, context_episode_env,
                     normalized_fitness, ACT_DIM, N_CONTEXT_CELLS)
from .networks import GaussianActor, flat_params, set_flat_params
from .sac import SACAgent, SACConfig
from .product_archive import ProductArchive, calibrate_probe_bd_ranges
from .qd_config import ContextualQDERLConfig


class _TorchActorProbePolicy:

    def __init__(self, actor: GaussianActor, to_tensor):
        self.actor = actor
        self.to_tensor = to_tensor

    def act(self, x):
        with torch.no_grad():
            _, _, mu = self.actor.sample(self.to_tensor(x))
        return mu.squeeze(0).cpu().numpy()

    def act_batch(self, X):
        device = next(self.actor.parameters()).device
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
            _, _, mu = self.actor.sample(Xt)
        return mu.cpu().numpy()


def rollout_contextual_torch(actor: GaussianActor, to_tensor, theta: np.ndarray,
                             scfg, n_zones: int, context_archive, warmstart_bank,
                             annual, load, rng: np.random.Generator,
                             n_context_cells: int, probe_grid,
                             price_profile: str | None = None):
    set_flat_params(actor, theta)
    cell, day, T_z0, T_m0, W_z0 = sample_training_context(
        context_archive, warmstart_bank, rng, n_context_cells)
    env, day_check = context_episode_env(annual, load, scfg, context_archive,
                                         cell, price_profile)
    assert day_check == day

    obs = env.reset(T_z0=T_z0, T_m0=T_m0, W_z0=W_z0)
    prev_obs = None
    F = 0.0
    done = False
    transitions = []
    while not done:
        x = augment_obs(obs, prev_obs, n_zones)
        with torch.no_grad():
            _, _, mu = actor.sample(to_tensor(x))
        a = mu.squeeze(0).cpu().numpy()
        next_obs, r, done, info = env.step(action_from_vector(a, scfg))
        x2 = augment_obs(next_obs, obs, n_zones)
        # Horizon timeout, not a terminal state -- bootstrap through it
        # (same convention as qd_erl.py / sac.py).
        transitions.append((x, a, r, x2, 0.0))
        prev_obs, obs = obs, next_obs
        F += r

    f_g36 = warmstart_bank.f_g36(day)
    F_tilde = normalized_fitness(F, f_g36)

    from ..probe_bd import probe_bd as _probe_bd
    bd = _probe_bd(_TorchActorProbePolicy(actor, to_tensor), scfg, grid=probe_grid)
    return float(F_tilde), bd, cell, day, float(F), float(f_g36), transitions


class ContextualQDERL:
    def __init__(self, qcfg: ContextualQDERLConfig = ContextualQDERLConfig(),
                seed: int = 0, mos_path: str | None = None,
                price_profile: str | None = None, n_rep_days: int | None = None):
        """`n_rep_days` is accepted and ignored (kept for call-site symmetry
        with `QDERL.__init__` / `run_qd_erl`) -- the §8.6 training unit is one
        real day per rollout (§8.6.2), not an `n_rep_days`-sized representative
        set; the representative set is still used elsewhere (§1.8 dispatch)."""
        torch.manual_seed(seed); np.random.seed(seed)
        self.q = qcfg
        self.rng = np.random.default_rng(seed)

        from ..scenarios import get_annual_context, get_context_archive, get_warmstart_bank
        ctx = get_annual_context(mos_path=mos_path)
        self.scfg = ctx.cfg
        self.annual, self.annual_load = ctx.annual, ctx.load
        self.context_archive = get_context_archive()
        self.warmstart_bank = get_warmstart_bank(mos_path=mos_path,
                                                 price_profile=price_profile)
        self.price_profile = price_profile
        self.mos_path = mos_path   # kept for RolloutPool construction in run()
        self.n_zones = len(self.scfg.zones)
        feat_dim = feature_dim_for(self.scfg)
        self.feat_dim = feat_dim

        sac_cfg = SACConfig(hidden=qcfg.hidden, gamma=qcfg.gamma, tau=qcfg.tau,
                            prefer_gpu=qcfg.prefer_gpu)
        self.agent = SACAgent(feat_dim, ACT_DIM, sac_cfg)   # shared critics + RL actor
        self.device = self.agent.device
        from .replay import ReplayBuffer
        self.replay = ReplayBuffer(sac_cfg.replay_capacity, feat_dim, ACT_DIM)

        # genome = flat params of a GaussianActor(feat_dim->4)
        self._actor_tmpl = GaussianActor(feat_dim, ACT_DIM, qcfg.hidden).to(self.device)
        self.n_params = flat_params(self._actor_tmpl).size
        self.iso_sigma = qcfg.effective_iso_sigma(self.n_params)
        print(f"  [ContextualQDERL] genome n_theta={self.n_params:,} -> "
              f"iso_sigma={self.iso_sigma:.4f}")

        from ..probe_bd import build_probe_grid
        self._probe_grid = build_probe_grid(self.scfg, self.n_zones,
                                            n_probe=qcfg.n_probe, seed=seed)

        bd_ranges = calibrate_probe_bd_ranges(self.scfg,
                                              n_genomes=qcfg.bd_calibration_genomes,
                                              hidden=qcfg.hidden, seed=seed)
        self.archive = ProductArchive(solution_dim=self.n_params, bd_ranges=bd_ranges,
                                      n_context_cells=qcfg.n_context_cells,
                                      n_beh=qcfg.n_beh, seed=seed)

        self.env_steps = 0
        self.grad_steps = 0
        self.generation = 0
        self.seed = seed
        self.parent = ""
        self.best_eval = -np.inf
        self._pool = None   # set in run() when qcfg.n_workers > 1 (§8.6.4e)

    # ------------------------------------------------------------------ #
    def _rollout(self, theta: np.ndarray, logger=None, generation: int = 0,
                source: str = ""):
        F_tilde, bd, cell, day, F_raw, f_g36, transitions = rollout_contextual_torch(
            self._actor_tmpl, self.agent.to_tensor, theta, self.scfg, self.n_zones,
            self.context_archive, self.warmstart_bank, self.annual, self.annual_load,
            self.rng, self.q.n_context_cells, self._probe_grid, self.price_profile)
        for x, a, r, x2, done in transitions:
            self.replay.add(x, a, r, x2, done)
        self.env_steps += len(transitions)
        if logger is not None:
            logger.log_rollout(generation=generation, env_steps=self.env_steps,
                               context_cell=cell, day=day, fitness_raw=F_raw,
                               f_g36=f_g36, fitness_tilde=F_tilde, source=source)
        return F_tilde, bd, cell

    def _rollout_batch(self, thetas, generation: int, logger=None, sources=None):
        if not thetas:
            return []
        sources = list(sources) if sources is not None else [""] * len(thetas)
        if self._pool is None:
            return [self._rollout(th, logger=logger, generation=generation, source=s)
                   for th, s in zip(thetas, sources)]
        out = []
        for r, s in zip(self._pool.submit_batch(thetas, generation), sources):
            for x, a, rw, x2, done in r["transitions"]:
                self.replay.add(x, a, rw, x2, done)
            self.env_steps += len(r["transitions"])
            if logger is not None:
                logger.log_rollout(generation=r["generation"], env_steps=self.env_steps,
                                   context_cell=r["cell"], day=r["day"],
                                   fitness_raw=r["F_raw"], f_g36=r["f_g36"],
                                   fitness_tilde=r["F_tilde"], source=s)
            out.append((r["F_tilde"], r["bd"], r["cell"]))
        return out

    # ---- evolutionary operator #
    def _iso_line(self):
        elites = self.archive.all_solutions()
        if len(elites) < 2:
            return np.random.randn(self.n_params) * 0.1, "random_fallback"
        i, j = np.random.randint(0, len(elites), 2)
        theta = (elites[i] + self.iso_sigma * np.random.randn(self.n_params)
                + self.q.line_sigma * (elites[j] - elites[i]) * np.random.randn())
        return theta, "iso_line"

    # ---- policy-gradient operator --------- #
    def _pg_variation(self, theta: np.ndarray):
        actor = GaussianActor(self.feat_dim, ACT_DIM, self.q.hidden).to(self.device)
        set_flat_params(actor, theta)
        opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        if len(self.replay) < 256:
            return theta
        for _ in range(self.q.n_grad_pg):
            batch = self.replay.sample(256, self.device)
            b = batch["obs"]
            u, logp, _ = actor.sample(b)
            q1, q2 = self.agent.critic(b, u)
            loss = (self.agent.alpha * logp - torch.min(q1, q2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return flat_params(actor)

    def _add(self, cell, theta, F, bd):
        self.archive.add(cell, theta, F, bd)

    # ---- resume (mirrors qd_erl.QDERL.load) ---------------------------- #
    def load(self, resume_from: str, logger=None, verbose: bool = True):
        """Restore nets/optimizers/archive/counters/RNG from a checkpoint."""
        from ..experiments.checkpoint import (resolve_checkpoint, load_agent,
                                              rebuild_product_archive)
        ptr = resolve_checkpoint(resume_from)
        agent, payload = load_agent(ptr["checkpoint"], prefer_gpu=self.q.prefer_gpu)
        self.agent = agent
        self.device = agent.device
        self.env_steps = int(payload.get("env_steps", 0))
        self.grad_steps = int(payload.get("grad_steps", 0))
        self.generation = int(payload.get("generation", 0))
        self.parent = ptr["checkpoint"]
        self.best_eval = float(payload.get("metrics", {}).get(
            "best_eval_return", -np.inf) or -np.inf)
        if ptr.get("archive_npz"):
            self.archive = rebuild_product_archive(ptr["archive_npz"], seed=self.seed)
        if logger is not None:
            logger.set_wall_offset(float(payload.get("wall_clock_s", 0.0)))
            logger.resumed_from = self.parent
        if verbose:
            print(f"  resumed from {self.parent} @ {self.env_steps:,} env steps, "
                  f"{self.archive.stats.num_elites} elites")
        return self

    # ------------------------------------------------------------------ #
    def run(self, generations: int = 100, total_env_steps: int | None = None,
           verbose: bool = True, logger=None, run_paths=None,
           checkpoint_every_gen: int = 0, log_every_gen: int = 10,
           archive_snapshot_gens=(), resume_from: str | None = None):
        q = self.q
        if resume_from:
            self.load(resume_from, logger=logger, verbose=verbose)

        if q.n_workers and q.n_workers > 1:
            from .rollout_pool import RolloutPool
            self._pool = RolloutPool(n_workers=q.n_workers, kind="torch",
                                     mos_path=self.mos_path,
                                     price_profile=self.price_profile,
                                     seed=self.seed, hidden=q.hidden,
                                     n_probe=q.n_probe,
                                     n_context_cells=q.n_context_cells,
                                     feat_dim=self.feat_dim, act_dim=ACT_DIM)
            if verbose:
                print(f"  [ContextualQDERL] rollout pool: {q.n_workers} "
                      f"CPU workers")
        try:
            if self.archive.stats.num_elites == 0:
                thetas = [np.random.randn(self.n_params) * q.init_sigma
                         for _ in range(q.n_init)]
                sources = ["bootstrap"] * len(thetas)
                for th, (F, bd, cell) in zip(
                        thetas, self._rollout_batch(thetas, 0, logger, sources)):
                    self._add(cell, th, F, bd)

            history = []
            gen_end = self.generation + generations
            while self.generation < gen_end:
                if total_env_steps and self.env_steps >= total_env_steps:
                    break
                self.generation += 1
                gen = self.generation

                # variation
                iso_pairs = [self._iso_line() for _ in range(q.g_ga)]
                cand = [t for t, s in iso_pairs]
                cand_src = [s for t, s in iso_pairs]
                base_sample = self.archive.sample_elites(q.g_pg, rng=self.rng)
                base = (list(base_sample["solution"]) if base_sample is not None
                       else [np.random.randn(self.n_params) * q.init_sigma
                            for _ in range(q.g_pg)])

                cand += [self._pg_variation(th) for th in base]
                cand_src += ["pg"] * len(base)

                # evaluation + insertion, feeding the shared buffer
                for th, (F, bd, cell) in zip(
                        cand, self._rollout_batch(cand, gen, logger, cand_src)):
                    self._add(cell, th, F, bd)

                last_losses = None
                for _ in range(q.n_grad_core):
                    if len(self.replay) >= 256:
                        last_losses = self.agent.update(self.replay)
                        self.grad_steps += 1

                # synergistic actor injection
                if gen % q.t_inj == 0:
                    th_rl = flat_params(self.agent.actor)
                    F, bd, cell = self._rollout(th_rl, logger=logger, generation=gen,
                                                source="actor_inject")
                    self._add(cell, th_rl, F, bd)

                # logging / checkpointing
                st = self.archive.stats
                if gen % log_every_gen == 0 or gen == 1:
                    ev = float(st.obj_max)
                    self.best_eval = max(self.best_eval, ev)
                    if logger is not None:
                        logger.log_train(env_steps=self.env_steps, generation=gen,
                                         grad_steps=self.grad_steps,
                                         best_return=st.obj_max, mean_return=st.obj_mean,
                                         eval_return=ev, coverage=st.coverage,
                                         qd_score=st.qd_score,
                                         archive_elites=st.num_elites,
                                         buffer_size=len(self.replay),
                                         alpha=float(self.agent.alpha),
                                         critic_loss=(last_losses or {}).get("critic"),
                                         actor_loss=(last_losses or {}).get("actor"))
                    elif verbose:
                        print(f"  gen {gen:4d} | steps {self.env_steps:>9,} | "
                              f"coverage {st.coverage:6.3f} | QD {st.qd_score:10.2f} | "
                              f"best F~ {st.obj_max:7.3f}")
                    history.append((gen, self.env_steps, st.coverage, st.qd_score,
                                   st.obj_max))
                if logger is not None and gen in tuple(archive_snapshot_gens):
                    logger.log_context_archive(self.archive, gen, self.env_steps)
                if run_paths is not None and checkpoint_every_gen and \
                        gen % checkpoint_every_gen == 0:
                    self._save(run_paths, logger, is_best=False)

            if run_paths is not None:
                if logger is not None:
                    logger.log_context_archive(self.archive, self.generation, self.env_steps)
                self._save(run_paths, logger, is_best=True)
            return self.archive, history
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

    def _save(self, run_paths, logger, is_best: bool = False):
        from ..experiments.checkpoint import save_checkpoint
        st = self.archive.stats
        return save_checkpoint(
            run_paths, learner="QD-ERL-Context", seed=self.seed,
            env_steps=self.env_steps, generation=self.generation,
            grad_steps=self.grad_steps,
            wall_clock_s=logger.wall_clock if logger else 0.0,
            agent=self.agent, archive=self.archive, config=self.q,
            metrics={"best_eval_return": round(float(st.obj_max), 4),
                     "qd_score": round(float(st.qd_score), 3),
                     "coverage": round(st.coverage, 5),
                     "archive_elites": int(st.num_elites)},
            parent=self.parent, archive_kind="product", is_best=is_best)


def run_contextual_qd_erl(generations: int = 100,
                          qcfg: ContextualQDERLConfig = ContextualQDERLConfig(),
                          total_env_steps: int | None = None, logger=None,
                          run_paths=None, checkpoint_every_gen: int = 0,
                          log_every_gen: int = 10, archive_snapshot_gens=(),
                          resume_from: str | None = None, verbose: bool = True,
                          **kw):
    algo = ContextualQDERL(qcfg=qcfg, **kw)
    return algo.run(generations=generations, total_env_steps=total_env_steps,
                    logger=logger, run_paths=run_paths, verbose=verbose,
                    checkpoint_every_gen=checkpoint_every_gen,
                    log_every_gen=log_every_gen,
                    archive_snapshot_gens=archive_snapshot_gens,
                    resume_from=resume_from)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run_contextual_qd_erl(generations=args.generations,
                         qcfg=ContextualQDERLConfig(prefer_gpu=not args.cpu))
