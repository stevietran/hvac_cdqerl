"""SAC over the Markov feature vector (torch, GPU-aware).

Plain Soft Actor-Critic over `common.augment_obs(o_t, o_{t-1})`

Runs on GPU automatically when available (`get_device`), else CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..config import default_singapore_config
from ..environment import HVACPlantEnv
from ..weather import SingaporeWeather
from .common import (feature_dim_for, augment_obs, PolicyController, ACT_DIM,
                     action_from_vector,
                     gamma_for_horizon)
from .networks import GaussianActor, TwinCritic, get_device
from .replay import ReplayBuffer


@dataclass
class SACConfig:
    hidden: int = 64
    gamma: float = 0.98
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    start_steps: int = 1_000
    updates_per_step: int = 1
    prefer_gpu: bool = True


class SACAgent:
    def __init__(self, feat_dim: int, act_dim: int, cfg: SACConfig = SACConfig()):
        self.cfg = cfg
        self.device = get_device(cfg.prefer_gpu)
        d = self.device
        self.actor = GaussianActor(feat_dim, act_dim, cfg.hidden).to(d)
        self.critic = TwinCritic(feat_dim, act_dim, cfg.hidden).to(d)
        self.critic_t = TwinCritic(feat_dim, act_dim, cfg.hidden).to(d)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=d)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -float(act_dim)
        self.feat_dim, self.act_dim = feat_dim, act_dim

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # ---- acting -------------------------------------------------------- #
    def to_tensor(self, x_np):
        return torch.as_tensor(x_np, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

    @torch.no_grad()
    def act(self, x, deterministic=False):
        if not torch.is_tensor(x):
            x = self.to_tensor(x)
        u, _, mu = self.actor.sample(x)
        a = mu if deterministic else u
        return a.squeeze(0).cpu().numpy()

    # ---- learning ------------------------------------------------------ #
    def update(self, buf: ReplayBuffer):
        batch = buf.sample(self.cfg.batch_size, self.device)
        b = batch["obs"]
        with torch.no_grad():
            b_next = batch["next_obs"]
            u2, logp2, _ = self.actor.sample(b_next)
            q1t, q2t = self.critic_t(b_next, u2)
            q_next = torch.min(q1t, q2t) - self.alpha * logp2
            y = batch["rew"] + self.cfg.gamma * (1 - batch["done"]) * q_next
        q1, q2 = self.critic(b, batch["act"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        b_det = b
        u, logp, _ = self.actor.sample(b_det)
        q1p, q2p = self.critic(b_det, u)
        actor_loss = (self.alpha * logp - torch.min(q1p, q2p)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)
        return dict(critic=critic_loss.detach().item(), actor=actor_loss.detach().item(),
                    alpha=float(self.alpha))


class SACController:
    """Wrap a trained SACAgent as an env controller (deterministic at eval)."""
    def __init__(self, agent: SACAgent, cfg):
        self.agent = agent; self.cfg = cfg
        self.n_zones = len(cfg.zones)
        self._prev_obs = None

    def reset(self):
        self._prev_obs = None

    def act(self, obs, info) -> dict:
        x = augment_obs(obs, self._prev_obs, self.n_zones)
        self._prev_obs = np.asarray(obs, dtype=float).copy()
        a = self.agent.act(self.agent.to_tensor(x), deterministic=True)
        return action_from_vector(a, self.cfg)


DEFAULT_SAC_EPISODE = os.environ.get("HVAC_SAC_EPISODE", "context_set")
SAC_EPISODE_KINDS = ("context_set", "representative")


def build_env(day_type="typical_cool", price_profile=None,
              n_rep_days=None, use_mos: bool = True, episode: str | None = None):
    """Training/eval env from the real .mos weather.
    """
    episode = (episode or DEFAULT_SAC_EPISODE).lower()
    if episode not in SAC_EPISODE_KINDS:
        raise ValueError(f"unknown SAC episode kind {episode!r}; "
                         f"expected one of {SAC_EPISODE_KINDS}")
    if use_mos:
        try:
            if episode == "context_set":
                from ..scenarios import make_context_set_env
                env, ctx, _ = make_context_set_env(price_profile=price_profile)
            else:
                from ..scenarios import make_training_env
                env, ctx, _ = make_training_env(n_rep_days=n_rep_days,
                                                price_profile=price_profile)
            return env, ctx.cfg
        except FileNotFoundError:
            pass
    cfg = default_singapore_config()
    return HVACPlantEnv(config=cfg,
                        weather=SingaporeWeather(day_type, price_profile)), cfg


@torch.no_grad()
def evaluate_agent(agent: SACAgent, scfg, day_type="typical_cool",
                   price_profile=None, n_rep_days=None, episode: str | None = None):
    """Deterministic evaluation episode -> (return, KPIs, trace).
    """
    from ..runner import run_episode
    env, cfg = build_env(day_type, price_profile, n_rep_days, episode=episode)
    ctrl = SACController(agent, cfg)
    kpis, trace = run_episode(env, ctrl)
    ret = sum(t["reward"] for t in trace)
    return ret, kpis, trace


def train_sac(total_steps: int = 20_000, cfg: SACConfig = SACConfig(),
              day_type: str = "typical_cool", price_profile: str | None = None,
              log_every: int = 2000, seed: int = 0, verbose: bool = True,
              run_paths=None, logger=None, checkpoint_every: int = 0,
              resume_from: str | None = None, eval_on_log: bool = True,
              n_rep_days=None, episode: str | None = None):
    """Train SAC, optionally logging to CSV and checkpointing / resuming
    """
    torch.manual_seed(seed); np.random.seed(seed)
    episode = (episode or DEFAULT_SAC_EPISODE).lower()
    env, scfg = build_env(day_type, price_profile, n_rep_days, episode=episode)
    feat_dim = feature_dim_for(scfg)
    n_zones = len(scfg.zones)
    steps_per_ep = int(round(env.horizon_hours * 60.0 / scfg.control_step_min))
    if verbose:
        print(f"  episode '{episode}': horizon {env.horizon_hours:.0f} h "
              f"({env.horizon_hours/24:.0f} days, {steps_per_ep:,} control steps) "
              f"-> {total_steps // max(steps_per_ep, 1):,} episodes at "
              f"{total_steps:,} steps")

    start_step, grad_steps, parent = 0, 0, ""
    best_eval = -np.inf
    if resume_from:
        from ..experiments.checkpoint import load_agent, resolve_checkpoint
        ptr = resolve_checkpoint(resume_from)
        agent, payload = load_agent(ptr["checkpoint"], prefer_gpu=cfg.prefer_gpu)
        start_step = int(payload.get("env_steps", 0))
        grad_steps = int(payload.get("grad_steps", 0))
        parent = ptr["checkpoint"]
        best_eval = float(payload.get("metrics", {}).get("best_eval_return", -np.inf))
        if logger is not None:
            logger.set_wall_offset(float(payload.get("wall_clock_s", 0.0)))
            logger.resumed_from = parent
        if verbose:
            print(f"  resumed from {parent} @ {start_step:,} env steps")
    else:
        agent = SACAgent(feat_dim, ACT_DIM, cfg)

    buf = ReplayBuffer(cfg.replay_capacity, feat_dim, ACT_DIM)
    warmup_until = start_step + cfg.start_steps

    obs = env.reset()
    prev_obs = None
    ep_ret, ep_rets = 0.0, []
    last = {}
    for step in range(start_step + 1, total_steps + 1):
        x = augment_obs(obs, prev_obs, n_zones)
        if step < warmup_until:
            a = np.random.uniform(-1, 1, ACT_DIM)
        else:
            a = agent.act(x, deterministic=False)

        next_obs, r, done, info = env.step(action_from_vector(a, scfg))
        x2 = augment_obs(next_obs, obs, n_zones)
        # TIME LIMIT != TERMINAL STATE
        timeout = bool(done)          # this env has no true terminal state
        buf.add(x, a, r, x2, 0.0 if timeout else float(done))
        prev_obs, obs, ep_ret = obs, next_obs, ep_ret + r
        if done:
            ep_rets.append(ep_ret); ep_ret = 0.0
            obs = env.reset(); prev_obs = None
        if step >= warmup_until and len(buf) >= cfg.batch_size:
            for _ in range(cfg.updates_per_step):
                last = agent.update(buf)
                grad_steps += 1

        if step % log_every == 0:
            ev = None
            if eval_on_log:
                ev, _, _ = evaluate_agent(agent, scfg, day_type, price_profile,
                                          n_rep_days, episode=episode)
            if logger is not None:
                logger.log_train(env_steps=step, grad_steps=grad_steps,
                                 best_return=max(ep_rets) if ep_rets else None,
                                 mean_return=float(np.mean(ep_rets[-5:])) if ep_rets else None,
                                 eval_return=ev, buffer_size=len(buf),
                                 alpha=float(agent.alpha), critic_loss=last.get("critic"),
                                 actor_loss=last.get("actor"))
            elif verbose:
                recent = np.mean(ep_rets[-5:]) if ep_rets else float("nan")
                print(f"  step {step:6d} | ep_return {recent:9.2f} | "
                      f"eval {ev if ev is not None else float('nan'):9.2f} | "
                      f"alpha {agent.alpha.item():.3f} | device {agent.device}")
            if ev is not None and ev > best_eval:
                best_eval = ev
                if run_paths is not None and checkpoint_every:
                    _save(run_paths, agent, cfg, seed, step, grad_steps, logger,
                          best_eval, parent, is_best=True)

        if run_paths is not None and checkpoint_every and step % checkpoint_every == 0:
            _save(run_paths, agent, cfg, seed, step, grad_steps, logger,
                  best_eval, parent, is_best=False)

    if run_paths is not None:
        _save(run_paths, agent, cfg, seed, total_steps, grad_steps, logger,
              best_eval, parent, is_best=False)
    return agent, scfg, ep_rets


def _save(run_paths, agent, cfg, seed, env_steps, grad_steps, logger,
          best_eval, parent, is_best):
    from ..experiments.checkpoint import save_checkpoint
    save_checkpoint(run_paths, learner="SAC", seed=seed, env_steps=env_steps,
                    grad_steps=grad_steps,
                    wall_clock_s=logger.wall_clock if logger else 0.0,
                    agent=agent, archive=None, config=cfg,
                    metrics={"best_eval_return": None if best_eval == -np.inf
                             else round(float(best_eval), 3)},
                    parent=parent, is_best=is_best)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train_sac(total_steps=args.steps, cfg=SACConfig(prefer_gpu=not args.cpu))
