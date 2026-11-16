"""Train zone-level RL rebalancing policies: behaviour-cloning warm start, then A2C fine-tuning.

Warm start: imitate the robust-LP controller (its post-move supply distribution for the bi-level
policy; its neighbour flow fractions for the mean-field policy). A2C then optimises episode profit
(fares minus empty-km cost) on training days; the checkpoint with the best validation-day profit
is kept, so fine-tuning can never end below the warm start on validation data.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..control.baselines import ControlContext, Controller, LPRebalance
from ..evaluation.metrics import episode_metrics
from ..ingestion.dataset import CityData
from ..simulation.fleet_sim import FleetSimTensor
from .policies import BilevelPolicy, MeanFieldPolicy

N_FEATURES = 12
K_NEIGHBORS = 8


def build_policy(kind: str, city: CityData, cfg: dict):
    r, g = cfg["control"]["rl"], city.graph
    if kind in ("bilevel_rl", "residual_rl"):
        pol = BilevelPolicy(N_FEATURES, r["hidden"], g.edge_index, g.edge_weight, r["concentration_scale"])
        if kind == "residual_rl":
            # Zero output layer -> equal concentrations -> mean correction 1/N per zone, i.e. exactly robust LP.
            torch.nn.init.zeros_(pol.actor[-1].weight)
            torch.nn.init.zeros_(pol.actor[-1].bias)
        return pol
    nb = torch.as_tensor(g.neighbor_table(K_NEIGHBORS), dtype=torch.long)
    dist = torch.as_tensor(g.dist_km, dtype=torch.float32)
    dist_feat = torch.gather(dist, 1, nb) / dist.mean()
    dist_feat[:, 0] = 0.0
    return MeanFieldPolicy(N_FEATURES, r["hidden"], g.edge_index, g.edge_weight, nb, dist_feat, r["concentration_scale"])


class PolicyController(Controller):
    def __init__(self, ctx: ControlContext, policy, kind: str, deterministic: bool = True):
        super().__init__(ctx)
        self.policy, self.kind, self.name, self.deterministic = policy, kind, kind, deterministic
        self.log_probs, self.values, self.entropies = [], [], []

    def _act(self, obs):
        dist, value = self.policy(self.ctx.features(obs))
        if self.deterministic:
            a = dist.mean
        else:
            a = dist.sample().clamp_min(1e-6)
            a = a / a.sum(-1, keepdim=True)
            lp, ent = dist.log_prob(a), dist.entropy()
            if lp.dim() == 2:                       # mean-field: independent per zone
                lp, ent = lp.sum(-1), ent.sum(-1)
            self.log_probs.append(lp)
            self.entropies.append(ent)
            self.values.append(value)
        return self.to_flows(a.detach(), obs)

    def to_flows(self, a: torch.Tensor, obs: dict) -> torch.Tensor:
        idle, arr = obs["idle"], obs["incoming"][:, 0]
        if self.kind == "residual_rl":
            # Residual on robust LP: rescale LP's post-move supply per zone by a * N (1 = keep LP's plan).
            lp = self._solve_all(obs, self.ctx.forecast(obs["t"], "hi")[:, 0] + obs["queue"].numpy())
            post = (idle - lp.sum(-1) + lp.sum(1) + arr).clamp_min(0)
            w = post * a * a.shape[-1]
            total = (idle.sum(-1) + arr.sum(-1))[:, None]
            return self._solve_all(obs, (w / w.sum(-1, keepdim=True).clamp_min(1e-9) * total).numpy())
        if self.kind == "bilevel_rl":
            total = (idle.sum(-1) + arr.sum(-1))[:, None]
            return self._solve_all(obs, (a * total).numpy())
        nb = self.policy.nb
        s, n, k = a.shape
        flows = torch.zeros(s, n, n).scatter_add_(2, nb.expand(s, n, k), idle[..., None] * a)
        flows[:, torch.arange(n), torch.arange(n)] = 0
        budget = self.ctx.cfg["sim"]["rebalance_budget"] * idle.sum(-1)
        moved = flows.sum((1, 2))
        return flows * torch.where(moved > budget, budget / moved.clamp_min(1e-9), torch.ones_like(moved))[:, None, None]


# ------------------------------------------------------------------ behaviour cloning
def _bc_targets(kind: str, obs: dict, flows: torch.Tensor, nb: torch.Tensor | None) -> torch.Tensor:
    idle, arr = obs["idle"], obs["incoming"][:, 0]
    if kind == "bilevel_rl":
        post = idle - flows.sum(-1) + flows.sum(1) + arr
        tgt = post.clamp_min(1e-3)
    else:
        s, n, k = idle.shape[0], idle.shape[1], nb.shape[1]
        moved = torch.gather(flows, 2, nb.expand(s, n, k))
        moved[:, :, 0] = 0
        frac = moved / idle.clamp_min(1e-6)[..., None]
        frac[:, :, 0] = (1 - frac[:, :, 1:].sum(-1)).clamp_min(0)
        tgt = frac.clamp_min(1e-3)
    return tgt / tgt.sum(-1, keepdim=True)


@torch.no_grad()
def collect_bc_data(kind, city, cfg, forecasts, days, nb):
    ctx = ControlContext(city, cfg, forecasts, days, cfg["sim"]["fleet_size"])
    teacher = LPRebalance(ctx, "hi")
    sim = FleetSimTensor(city, cfg, days)
    obs, done, xs, ys = sim.reset(), False, [], []
    while not done:
        flows = teacher.act(obs)
        xs.append(ctx.features(obs))
        ys.append(_bc_targets(kind, obs, flows, nb))
        obs, _, done = sim.step(flows)
    return torch.cat(xs), torch.cat(ys)


def behaviour_clone(policy, x, y, epochs, lr, log):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            dist, _ = policy(x[idx])
            loss = -dist.log_prob(y[idx]).mean() / y.shape[1]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 25 == 0:
            log(f"  BC epoch {ep + 1:3d}  nll/zone {total / len(x):.4f}")


# ------------------------------------------------------------------ evaluation + A2C
@torch.no_grad()
def evaluate_policy(policy, kind, city, cfg, forecasts, days) -> float:
    ctx = ControlContext(city, cfg, forecasts, days, cfg["sim"]["fleet_size"])
    ctrl = PolicyController(ctx, policy, kind, deterministic=True)
    sim = FleetSimTensor(city, cfg, days)
    obs, done = sim.reset(), False
    while not done:
        obs, _, done = sim.step(ctrl.act(obs))
    return float(np.mean([episode_metrics(t)["profit_AED"] for t in sim.episode_totals()]))


def train_policy(kind: str, city: CityData, cfg: dict, log=print, scenarios_per_episode: int = 8):
    from ..evaluation.run_eval import load_forecasts
    r = cfg["control"]["rl"]
    rng = np.random.default_rng(cfg["seed"])
    forecasts = load_forecasts(cfg)
    skip = cfg["forecast"]["periodic_days"] + 1
    train_days = city.split_day_starts("train")[skip:]
    val_days = city.split_day_starts("val")
    policy = build_policy(kind, city, cfg)
    nb = getattr(policy, "nb", None)
    t0 = time.time()

    if kind == "residual_rl":
        log(f"[rl] {kind}: starts as robust LP (zero correction), no behaviour cloning needed")
    else:
        log(f"[rl] {kind}: behaviour cloning on {len(train_days)} training days")
        x, y = collect_bc_data(kind, city, cfg, forecasts, train_days, nb)
        behaviour_clone(policy, x, y, r["bc_epochs"], r["lr"], log)
    best_profit = evaluate_policy(policy, kind, city, cfg, forecasts, val_days)
    best_state = copy.deepcopy(policy.state_dict())
    log(f"  warm-start val profit {best_profit:,.0f} AED")

    opt = torch.optim.Adam(policy.parameters(), lr=r["lr"])
    reward_scale = None
    for ep in range(r["episodes"]):
        days = list(rng.choice(train_days, size=scenarios_per_episode, replace=False))
        ctx = ControlContext(city, cfg, forecasts, days, cfg["sim"]["fleet_size"])
        ctrl = PolicyController(ctx, policy, kind, deterministic=False)
        sim = FleetSimTensor(city, cfg, days)
        obs, done, rewards = sim.reset(), False, []
        while not done:
            obs, rew, done = sim.step(ctrl.act(obs))
            rewards.append(rew)
        rew = torch.stack(rewards)                                         # [T, S]
        reward_scale = reward_scale or float(rew.abs().mean())
        rew = rew / reward_scale
        logp, values, ent = torch.stack(ctrl.log_probs), torch.stack(ctrl.values), torch.stack(ctrl.entropies)
        # Generalised advantage estimation (episode ends at the operating-day cut-off).
        v = values.detach()
        adv, running = torch.zeros_like(rew), torch.zeros(rew.shape[1])
        for t in reversed(range(len(rew))):
            v_next = v[t + 1] if t + 1 < len(rew) else torch.zeros_like(running)
            running = rew[t] + r["gamma"] * v_next - v[t] + r["gamma"] * r["gae_lambda"] * running
            adv[t] = running
        returns = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        actor_on = float(ep >= r["critic_warmup"])          # fit the critic before moving the actor
        loss = actor_on * (-(logp * adv).mean() - r["entropy_coef"] * ent.mean()) + 0.5 * F.smooth_l1_loss(values, returns)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 25 == 0 or ep + 1 == r["episodes"]:
            val = evaluate_policy(policy, kind, city, cfg, forecasts, val_days)
            tag = ""
            if val > best_profit:
                best_profit, best_state, tag = val, copy.deepcopy(policy.state_dict()), "  *best*"
            log(f"  episode {ep + 1:4d}  train return {rew.sum(0).mean():.2f}  val profit {val:,.0f} AED{tag}"
                f"  ({time.time() - t0:.0f}s)")
    policy.load_state_dict(best_state)
    torch.save(best_state, cfg["paths"]["output_dir"] / f"policy_{kind}.pt")
    log(f"  saved policy_{kind}.pt (best val profit {best_profit:,.0f} AED)")
    return policy


def load_policy_controller(kind: str, ctx: ControlContext, cfg: dict) -> PolicyController:
    path = cfg["paths"]["output_dir"] / f"policy_{kind}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python main.py train-rl` first.")
    policy = build_policy(kind, ctx.city, cfg)
    policy.load_state_dict(torch.load(path, weights_only=True))
    policy.eval()
    return PolicyController(ctx, policy, kind, deterministic=True)
