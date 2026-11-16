"""Pipeline orchestrator.

    python main.py data        # generate / aggregate zone-level demand
    python main.py forecast    # train + benchmark forecasters, export forecasts with conformal intervals
    python main.py train-rl    # train the bi-level and mean-field rebalancing policies
    python main.py evaluate    # compare all control methods on held-out test days
    python main.py sweep       # fleet-size sweep (policies trained once, evaluated at each size)
    python main.py dfl         # decision-focused fine-tuning of the forecaster through the LP
    python main.py event       # vehicle-level validation: learn V(zone, slot), compare rebalancers x matchers
    python main.py stress      # Ramadan / concert / New Year's Eve stress tests
    python main.py transfer    # pretrain on NYC TLC, zero/few-shot transfer to the Dubai-like city
    python main.py sumo        # replay test-day demand in SUMO with TraCI-controlled taxis (--gui to watch)
    python main.py all         # everything above in order
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.utils import get_device, load_config, save_json, seed_everything


def cmd_data(cfg):
    from src.ingestion.dataset import prepare_data
    city = prepare_data(cfg, np.random.default_rng(cfg["seed"]))
    p = city.pickups
    spd = city.slots_per_day
    print(f"zones {city.graph.num_zones}  slots {len(city.slot_index)}  days {len(city.slot_index) // spd}  "
          f"split(days) {city.split_days}")
    print(f"trips/day {p.sum() / (len(p) / spd):,.0f}  peak trips/slot {p.sum(1).max():,.0f}")


def cmd_forecast(cfg, models):
    from src.evaluation.report import write_forecast_table
    from src.ingestion.dataset import load_data
    from src.models.train_forecast import run_forecasting
    city = load_data(cfg)
    names = models or ["last", "ha", "gcngru", "gcngru_ae", "agcrn", "gwnet", "staeformer"]
    if cfg["forecast"]["model"] not in names:
        names.append(cfg["forecast"]["model"])
    res = run_forecasting(city, cfg, get_device(cfg), names)
    save_json(res, cfg["paths"]["output_dir"] / "forecast_metrics.json")
    print(write_forecast_table(res, cfg["paths"]["output_dir"] / "forecast_metrics.md"))


def cmd_train_rl(cfg, methods):
    from src.ingestion.dataset import load_data
    from src.models.train_rl import train_policy
    city = load_data(cfg)
    for kind in methods or ["bilevel_rl", "meanfield_rl"]:
        train_policy(kind, city, cfg)


def cmd_evaluate(cfg, methods, fleet=None, tag=""):
    from src.evaluation.report import write_control_table
    from src.evaluation.run_eval import evaluate_methods
    from src.ingestion.dataset import load_data
    city = load_data(cfg)
    if fleet:
        cfg["sim"]["fleet_size"] = fleet
    res = evaluate_methods(city, cfg, methods or cfg["control"]["methods"])
    out = cfg["paths"]["output_dir"]
    save_json(res, out / f"control_metrics{tag}.json")
    print(write_control_table(res, out / f"control_metrics{tag}.md",
                              title=f"Fleet size {cfg['sim']['fleet_size']}, test days"))
    return res


def cmd_sweep(cfg, methods):
    from src.evaluation.report import write_sweep_table
    results = {}
    for n in cfg["eval"]["fleet_sweep"]:
        print(f"\n=== fleet {n} ===")
        results[n] = cmd_evaluate(cfg, methods, fleet=n, tag=f"_fleet{n}")
    print(write_sweep_table(results, cfg["paths"]["output_dir"] / "fleet_sweep.md"))


def cmd_dfl(cfg):
    import torch
    from src.evaluation.metrics import forecast_metrics
    from src.evaluation.report import write_control_table, write_forecast_table
    from src.evaluation.run_eval import evaluate_methods, load_forecasts
    from src.ingestion.dataset import load_data
    from src.models.conformal import SplitConformal
    from src.models.decision_focused import train_decision_focused
    from src.models.train_forecast import export_forecasts, load_trained_forecaster, predict
    city, device = load_data(cfg), get_device(cfg)
    name = cfg["forecast"]["model"]
    base, base_conf, _, ds = load_trained_forecaster(name, city, cfg, device)
    base_pred = predict(base, ds["test"], device)
    model, ds, scaler = train_decision_focused(base, city, cfg, load_forecasts(cfg), device)
    out = cfg["paths"]["output_dir"]
    torch.save(model.state_dict(), out / f"forecaster_{name}_dfl.pt")
    conf = SplitConformal(cfg["forecast"]["conformal_alpha"]).calibrate(predict(model, ds["val"], device), ds["val"].y)
    export_forecasts(city, cfg, model, conf, scaler, device, filename="forecasts_dfl.npz")
    pred = predict(model, ds["test"], device)
    fres = {name: forecast_metrics(base_pred, ds["test"].y, *base_conf.interval(base_pred)),
            f"{name}_dfl": forecast_metrics(pred, ds["test"].y, *conf.interval(pred))}
    bias = {k: float((p[..., 0] - ds["test"].y[..., 0]).mean()) for k, p in ((name, base_pred), (f"{name}_dfl", pred))}
    text = write_forecast_table(fres, out / "dfl_forecast_metrics.md")
    text += "\nMean pickup forecast bias (pred - actual): " + ", ".join(f"{k} {v:+.3f}" for k, v in bias.items()) + "\n"
    (out / "dfl_forecast_metrics.md").write_text(text, encoding="utf-8")
    print(text)
    res = evaluate_methods(city, cfg, ["lp", "lp_dfl", "lp_robust", "lp_robust_dfl"])
    save_json(res, out / "dfl_control_metrics.json")
    print(write_control_table(res, out / "dfl_control_metrics.md", title="MAE-trained vs decision-focused forecasts"))


def cmd_event(cfg, methods):
    from src.evaluation.event_eval import evaluate_event, learn_value_table
    from src.evaluation.report import write_control_table
    from src.ingestion.dataset import load_data
    city = load_data(cfg)
    learn_value_table(city, cfg, n_days=10)
    res = evaluate_event(city, cfg, methods or ["none", "lp", "lp_robust", "bilevel_rl"], cfg["event_sim"]["matchers"])
    out = cfg["paths"]["output_dir"]
    save_json(res, out / "event_metrics.json")
    print(write_control_table(res, out / "event_metrics.md",
                              title=f"vehicle-level simulator, {cfg['event_sim']['eval_days']} test days, rebalancer + matcher"))


def cmd_stress(cfg, methods):
    from src.evaluation.stress import run_stress, write_stress_tables
    from src.ingestion.dataset import load_data
    res = run_stress(load_data(cfg), cfg, methods or ["none", "lp", "lp_robust", "bilevel_rl"],
                     ["ha", "gcngru_ae", "staeformer"])
    save_json(res, cfg["paths"]["output_dir"] / "stress_metrics.json")
    print(write_stress_tables(res, cfg["paths"]["output_dir"] / "stress_metrics.md"))


def cmd_transfer(cfg, methods=None):
    """Own NYC-pretrained transfer study plus published OpenCity zero-shot rows.

    `--methods opencity` recomputes only the OpenCity rows and merges them into existing results.
    """
    import copy
    import json
    from src.evaluation.report import write_transfer_table
    from src.ingestion.dataset import load_data, prepare_data
    from src.models.opencity_adapter import WEIGHTS, opencity_zero_shot
    from src.models.transfer import run_transfer
    target, device = load_data(cfg), get_device(cfg)
    path = cfg["paths"]["output_dir"] / "transfer_metrics.json"
    if methods == ["opencity"] and path.exists():
        res = json.loads(path.read_text())
    else:
        src_cfg = copy.deepcopy(cfg)
        src_cfg["data"]["source"] = "nyc_tlc"
        try:
            source = load_data(src_cfg)
        except FileNotFoundError:
            source = prepare_data(src_cfg, np.random.default_rng(cfg["seed"]))
        res = run_transfer(source, target, cfg, device)
    for variant in ("base", "plus"):
        if (WEIGHTS / f"OpenCity-{variant}.pth").exists():
            res[f"OpenCity-{variant} zero-shot (published weights)"] = opencity_zero_shot(target, cfg, variant, device)
            print(f"  OpenCity-{variant}: test MAE {res[f'OpenCity-{variant} zero-shot (published weights)']['all']['MAE']:.3f}")
    save_json(res, path)
    print(write_transfer_table(res, cfg["paths"]["output_dir"] / "transfer_metrics.md"))


def cmd_sumo(cfg, methods, gui=False, fleet=None, replay=False, ablation=False):
    if ablation:                                  # 5-min decisions, travel-time LP, value dispatch: one at a time
        from src.simulation.sumo_bridge import run_sumo_ablation
        run_sumo_ablation(cfg, (methods or ["lp_robust"])[0])
        return
    if replay:                                    # record 1 h on the Dubai map for the satellite viewer
        import shutil
        from pathlib import Path
        from src.simulation.sumo_bridge import run_sumo_episode
        cfg["sumo"]["network"] = "osm"
        out = cfg["paths"]["output_dir"] / "sumo_viewer"
        for m in methods or ["residual_rl", "lp_robust", "bilevel_rl", "none"]:
            run_sumo_episode(cfg, m, fleet=fleet or 200, slots=4, record_dir=out)
        shutil.copy(Path("src/simulation/viewer.html"), out / "index.html")
        shutil.copy(Path("src/simulation/director.js"), out / "director.js")
        print(f"open {out / 'index.html'} in a browser")
        return
    if gui:                                       # one short live episode in sumo-gui, no reports written
        from src.simulation.sumo_bridge import run_sumo_episode
        run_sumo_episode(cfg, (methods or ["lp_robust"])[0], fleet=fleet or 200, slots=4, gui=True)
        return
    from src.simulation.sumo_bridge import run_sumo_comparison
    run_sumo_comparison(cfg, methods or ["none", "lp", "lp_robust"])


COMMANDS = ["data", "forecast", "train-rl", "evaluate", "sweep", "dfl", "event", "stress", "transfer", "sumo", "all"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", nargs="*", help="forecasters, RL policies or control methods to run")
    ap.add_argument("--fleet", type=int, help="override sim.fleet_size for evaluate; taxi count for sumo --gui")
    ap.add_argument("--gui", action="store_true", help="sumo: watch one 1-hour episode live in sumo-gui")
    ap.add_argument("--network", choices=["grid", "osm"], help="sumo: synthetic grid or real Dubai roads (OSM)")
    ap.add_argument("--ablation", action="store_true", help="sumo: add the control upgrades one at a time")
    ap.add_argument("--replay", action="store_true", help="sumo: record runs on the Dubai map for the satellite viewer")
    ap.add_argument("--3d", dest="view3d", action="store_true", help="sumo --gui: also open a 3D view")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    if args.network:
        cfg["sumo"]["network"] = args.network
    if args.view3d:
        cfg["sumo"]["gui_3d"] = True
    t0 = time.time()

    def run(name):
        return args.command in (name, "all")

    def own(name):
        return args.methods if args.command == name else None

    if run("data"):
        cmd_data(cfg)
    if run("forecast"):
        cmd_forecast(cfg, own("forecast"))
    if run("train-rl"):
        cmd_train_rl(cfg, own("train-rl"))
    if run("evaluate"):
        cmd_evaluate(cfg, own("evaluate"), fleet=args.fleet)
    if run("sweep"):
        cmd_sweep(cfg, own("sweep"))
    if run("dfl"):
        cmd_dfl(cfg)
    if run("event"):
        cmd_event(cfg, own("event"))
    if run("stress"):
        cmd_stress(cfg, own("stress"))
    if run("transfer"):
        cmd_transfer(cfg, own("transfer"))
    if run("sumo"):
        cmd_sumo(cfg, own("sumo"), gui=args.gui, fleet=args.fleet, replay=args.replay, ablation=args.ablation)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
