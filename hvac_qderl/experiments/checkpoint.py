"""Versioned checkpointing, resume, and archive (de)serialisation.

Design notes
------------
* torch is imported lazily so that *loading an archive and dispatching / plotting
  works without torch at all* — only the SAC/QD-ERL network tensors need it.
* The pyribs archive is stored as a plain `.npz` (solutions, objectives,
  measures) plus its construction kwargs, so it can be rebuilt with any pyribs
  version and inspected without pyribs at all.
* `manifest.json` is human-readable provenance: config, counters, metrics, RNG
  state, versions, device, and the *parent* checkpoint so a resumed lineage is
  traceable
"""
from __future__ import annotations

import os
import platform
import random
import time
from dataclasses import asdict, is_dataclass

import numpy as np

from .logging_utils import RunPaths, write_json, read_json

SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------- #
def capture_rng() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch
        st["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            st["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return st


def restore_rng(st: dict):
    if not st:
        return
    try:
        if "python" in st:
            random.setstate(_tuplize(st["python"]))
        if "numpy" in st:
            np.random.set_state(_tuplize(st["numpy"]))
    except Exception:
        pass
    try:
        import torch
        if "torch" in st and st["torch"] is not None:
            torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu")
                                else st["torch"])
        if "torch_cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["torch_cuda"])
    except Exception:
        pass


def _tuplize(x):
    """RNG states round-trip through torch.save as lists; make them tuples again."""
    if isinstance(x, list):
        return tuple(_tuplize(v) for v in x)
    return x


def _cfg_to_dict(cfg):
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return {k: v for k, v in asdict(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
    if isinstance(cfg, dict):
        return cfg
    return {k: v for k, v in vars(cfg).items()
            if isinstance(v, (int, float, str, bool, type(None)))}


# --------------------------------------------------------------------------- #
# Archive serialisation (pyribs-version-independent)
# --------------------------------------------------------------------------- #
def save_archive(archive, path: str, build_kwargs: dict | None = None):
    if archive is None:
        return None
    d = archive.data(["solution", "objective", "measures", "index"])
    np.savez_compressed(
        path,
        solution=np.asarray(d["solution"], dtype=np.float32),
        objective=np.asarray(d["objective"], dtype=np.float64),
        measures=np.asarray(d["measures"], dtype=np.float32),
        index=np.asarray(d["index"], dtype=np.int64),
        build_kwargs=np.array([repr(build_kwargs or {})], dtype=object),
        cells=np.int64(getattr(archive, "cells", 0)),
        solution_dim=np.int64(getattr(archive, "solution_dim", 0)),
    )
    return path


def load_archive_arrays(path: str) -> dict:
    """Load raw archive arrays — no pyribs required (used by dispatch & figures)."""
    z = np.load(path, allow_pickle=True)
    return dict(solution=z["solution"], objective=z["objective"],
                measures=z["measures"], index=z["index"],
                # optional explicit feasibility measure; older archives lack it
                rh_violation=z["rh_violation"] if "rh_violation" in z else None,
                cells=int(z["cells"]) if "cells" in z else 0,
                solution_dim=int(z["solution_dim"]) if "solution_dim" in z else 0)


# --------------------------------------------------------------------------- #
# product archive serialisation -- a SEPARATE pair of functions from
# save_archive/rebuild_archive above, not a repurposing: `ProductArchive`
# --------------------------------------------------------------------------- #
def save_product_archive(archive, path: str, build_kwargs: dict | None = None):
    if archive is None:
        return None
    a = archive.to_arrays()
    lo = np.array([r[0] for r in archive.bd_ranges], dtype=np.float64)
    hi = np.array([r[1] for r in archive.bd_ranges], dtype=np.float64)
    np.savez_compressed(
        path,
        solution=a["solution"], objective=a["objective"], measures=a["measures"],
        index=a["index"], context_cell=a["context_cell"],
        bd_ranges_lo=lo, bd_ranges_hi=hi,
        n_context_cells=np.int64(archive.n_context_cells),
        n_beh=np.int64(archive.n_beh),
        solution_dim=np.int64(archive.solution_dim),
        build_kwargs=np.array([repr(build_kwargs or {})], dtype=object),
    )
    return path


def load_product_archive_arrays(path: str) -> dict:
    """Load raw §8.6 product-archive arrays -- no pyribs required (mirrors
    `load_archive_arrays` above)."""
    z = np.load(path, allow_pickle=True)
    bd_ranges = list(zip(z["bd_ranges_lo"].tolist(), z["bd_ranges_hi"].tolist()))
    return dict(solution=z["solution"], objective=z["objective"],
               measures=z["measures"], index=z["index"],
               context_cell=z["context_cell"], bd_ranges=bd_ranges,
               n_context_cells=int(z["n_context_cells"]), n_beh=int(z["n_beh"]),
               solution_dim=int(z["solution_dim"]))


def rebuild_product_archive(path: str, seed: int = 0,
                            qd_score_offset: float = -10.0):
    """Rebuild a live `ProductArchive` and re-insert the stored elites (for
    resume) -- the §8.6 analogue of `rebuild_archive` above."""
    if not path:
        raise ValueError(
            "Cannot rebuild ProductArchive: archive path is missing. "
            "Pass a run directory or checkpoint containing a saved archive_npz.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Product archive file not found: {path}")
    from ..learners.product_archive import ProductArchive
    a = load_product_archive_arrays(path)
    archive = ProductArchive.from_arrays(
        a, solution_dim=a["solution_dim"], bd_ranges=a["bd_ranges"],
        n_context_cells=a["n_context_cells"], n_beh=a["n_beh"], seed=seed,
        qd_score_offset=qd_score_offset)
    return archive

# --------------------------------------------------------------------------- #
# Checkpoint save / load
# --------------------------------------------------------------------------- #
def save_checkpoint(paths: RunPaths, *, learner: str, seed: int, env_steps: int,
                    generation: int = 0, grad_steps: int = 0,
                    wall_clock_s: float = 0.0, agent=None, archive=None,
                    config=None, metrics: dict | None = None,
                    parent: str = "", archive_kind: str = "cvt",
                    is_best: bool = False) -> str:
    """Write ckpt_<env_steps>.pt (+ archive npz) and update manifest/pointers."""
    ckpt_path = paths.ckpt(env_steps)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "learner": learner, "seed": seed, "env_steps": env_steps,
        "generation": generation, "grad_steps": grad_steps,
        "wall_clock_s": wall_clock_s,
        "config": _cfg_to_dict(config),
        "metrics": metrics or {},
        "rng": capture_rng(),
        "parent": parent,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if agent is not None:
        payload["feat_dim"] = getattr(agent, "feat_dim", None)
        payload["act_dim"] = getattr(agent, "act_dim", None)
        payload["device"] = str(getattr(agent, "device", "cpu"))
        payload["nets"] = {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_t": agent.critic_t.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
        }
        payload["optims"] = {
            "critic_opt": agent.critic_opt.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
        }

    arch_path = None
    if archive is not None:
        # ProductArchive (18 CVTArchives, product_archive.py) needs
        # its own serialisation -- save_archive assumes a single pyribs Archive
        from ..learners.product_archive import ProductArchive
        if isinstance(archive, ProductArchive):
            arch_path = save_product_archive(archive, paths.archive_npz(env_steps),
                                             {"kind": archive_kind})
        else:
            arch_path = save_archive(archive, paths.archive_npz(env_steps),
                                     {"kind": archive_kind,
                                      "cells": getattr(archive, "cells", None)})
        payload["archive_npz"] = os.path.basename(arch_path)
        payload["archive_kind"] = archive_kind

    try:
        import torch
        torch.save(payload, ckpt_path)
    except ImportError:
        # torch absent: persist the non-tensor payload so archives/manifests still work
        import pickle
        ckpt_path = ckpt_path.replace(".pt", ".pkl")
        payload.pop("nets", None); payload.pop("optims", None)
        with open(ckpt_path, "wb") as f:
            pickle.dump(payload, f)

    manifest = {k: v for k, v in payload.items() if k not in ("nets", "optims", "rng")}
    manifest.update(checkpoint=os.path.basename(ckpt_path),
                    archive_npz=os.path.basename(arch_path) if arch_path else None,
                    torch_version=_torch_version(), numpy_version=np.__version__,
                    python=platform.python_version(), platform=platform.platform())
    write_json(paths.manifest, manifest)
    write_json(paths.latest_ptr, {"checkpoint": ckpt_path,
                                  "archive_npz": arch_path,
                                  "env_steps": env_steps})
    if is_best:
        write_json(paths.best_ptr, {"checkpoint": ckpt_path,
                                    "archive_npz": arch_path,
                                    "env_steps": env_steps,
                                    "metrics": metrics or {}})
    return ckpt_path


def load_payload(ckpt_path: str) -> dict:
    if ckpt_path.endswith(".pkl"):
        import pickle
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    import torch
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:      # older torch without weights_only
        return torch.load(ckpt_path, map_location="cpu")


def _reanchor(ptr: dict, json_path: str) -> dict:
    """Re-point a pointer file's paths at the directory it was found in.
    Take the trailing component regardless of separator and look for it next to
    the pointer file. Fall back to the stored path when that fails, so a layout
    this function does not anticipate still behaves as before.
    """
    import re
    root = os.path.dirname(os.path.abspath(json_path))
    out = dict(ptr)
    for key in ("checkpoint", "archive_npz"):
        val = ptr.get(key)
        if not val or os.path.exists(val):
            continue
        base = re.split(r"[\\/]", str(val))[-1]
        for cand in (os.path.join(root, "checkpoints", base),
                     os.path.join(root, base)):
            if os.path.exists(cand):
                out[key] = cand
                break
    return out


def resolve_checkpoint(spec: str) -> dict:
    """Accept a run dir, a latest.json/best.json, or a checkpoint file."""
    if os.path.isdir(spec):
        for name in ("latest.json", "best.json"):
            p = os.path.join(spec, name)
            if os.path.exists(p):
                return _reanchor(read_json(p), p)
        raise FileNotFoundError(f"no latest.json/best.json in {spec}")
    if spec.endswith(".json"):
        return _reanchor(read_json(spec), spec)
    return {"checkpoint": spec,
            "archive_npz": _guess_archive(spec)}


def _guess_archive(ckpt_path: str):
    base = os.path.basename(ckpt_path)
    for ext in (".pt", ".pkl"):
        if base.endswith(ext):
            steps = base[len("ckpt_"):-len(ext)]
            cand = os.path.join(os.path.dirname(ckpt_path), f"archive_{steps}.npz")
            if os.path.exists(cand):
                return cand
            run_root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
            manifest_path = os.path.join(run_root, "manifest.json")
            if os.path.exists(manifest_path):
                manifest = read_json(manifest_path)
                manifest = _reanchor(manifest, manifest_path)
                archive = manifest.get("archive_npz")
                if archive and os.path.exists(archive):
                    return archive
            return None
    return None


def load_agent(ckpt_path: str, prefer_gpu: bool = True):
    """Rebuild a SACAgent with weights, optimizers and RNG restored (needs torch)."""
    payload = load_payload(ckpt_path)
    from ..learners.common import ACT_DIM, feature_dim_for
    from ..config import default_singapore_config
    saved_act = int(payload.get("act_dim", -1))
    if saved_act != ACT_DIM:
        raise ValueError(
            f"CHECKPOINT INVALID: saved with act_dim={saved_act}, current build "
            f"uses ACT_DIM={ACT_DIM}. History: the action space gained "
            f"`plant_enable` (2 -> 3), then an explicit staging head `n_stage`, "
            f"a[3] (3 -> 4), which was REMOVED (4 -> 3) as strictly dominated; "
            f"a[3] was then reused for `flow_cap` (3 -> 4) and that head is "
            f"currently DEFERRED (4 -> 3) -- see learners/common.py's ACT_DIM "
            f"block. Actor output layers and archive genomes therefore have "
            f"incompatible shapes. Re-train from scratch; migration is not "
            f"meaningful because a removed head's weights have no target to "
            f"map onto.")

    want_feat = feature_dim_for(default_singapore_config())
    saved_feat = payload.get("feat_dim", payload.get("obs_dim"))
    if "belief_dim" in payload.get("config", {}) or (
            saved_feat is not None and int(saved_feat) != int(want_feat)):
        raise ValueError(
            f"CHECKPOINT INVALID: saved with input width {saved_feat} "
            f"(belief_dim present: {'belief_dim' in payload.get('config', {})}); "
            f"current build feeds the actor a {want_feat}-D Markov feature "
            f"vector (obs 2n+21, plus a per-zone dT_z block) with no GRU "
            f"encoder. Re-train from scratch; zero-padding the first layer "
            f"would leave the newer channels (q_evap, plant utilisation, the "
            f"+1/+2/+3 h t_oa and t_wb forecasts, hours-until-occupancy, "
            f"dT_z) attached to untrained weights.")
    from ..learners.sac import SACAgent, SACConfig
    cfgd = payload.get("config", {})
    sac_cfg = SACConfig(**{k: v for k, v in cfgd.items()
                           if k in SACConfig.__dataclass_fields__})
    sac_cfg.prefer_gpu = prefer_gpu
    agent = SACAgent(int(saved_feat), payload["act_dim"], sac_cfg)
    nets = payload["nets"]
    agent.actor.load_state_dict(nets["actor"])
    agent.critic.load_state_dict(nets["critic"])
    agent.critic_t.load_state_dict(nets["critic_t"])
    with_no_grad_copy(agent, nets["log_alpha"])
    if "optims" in payload:
        try:
            agent.critic_opt.load_state_dict(payload["optims"]["critic_opt"])
            agent.actor_opt.load_state_dict(payload["optims"]["actor_opt"])
            agent.alpha_opt.load_state_dict(payload["optims"]["alpha_opt"])
        except Exception:
            pass
    restore_rng(payload.get("rng", {}))
    return agent, payload


def with_no_grad_copy(agent, log_alpha_tensor):
    import torch
    with torch.no_grad():
        agent.log_alpha.copy_(log_alpha_tensor.to(agent.log_alpha.device))


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def describe(ckpt_or_run: str) -> str:
    """Human-readable summary of a saved agent (the `--describe` CLI output)."""
    ptr = resolve_checkpoint(ckpt_or_run)
    ck = ptr["checkpoint"]
    run_dir = os.path.dirname(os.path.dirname(ck))
    man_path = os.path.join(run_dir, "manifest.json")
    man = read_json(man_path) if os.path.exists(man_path) else load_payload(ck)
    lines = [
        f"checkpoint     : {ck}",
        f"learner        : {man.get('learner')}  (seed {man.get('seed')})",
        f"env steps      : {man.get('env_steps'):,}" if man.get("env_steps") else "",
        f"generation     : {man.get('generation')}",
        f"grad steps     : {man.get('grad_steps')}",
        f"wall clock     : {man.get('wall_clock_s')} s",
        f"created        : {man.get('created')}",
        f"parent ckpt    : {man.get('parent') or '(none — fresh run)'}",
        f"archive        : {man.get('archive_npz') or '(none)'}",
        f"device/torch   : {man.get('device')} / torch {man.get('torch_version')}",
        f"metrics        : {man.get('metrics')}",
        f"config         : {man.get('config')}",
    ]
    return "\n".join(l for l in lines if l)
