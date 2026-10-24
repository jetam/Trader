"""Command line entry point: python -m trader <command>."""

import argparse
import logging
import time

import numpy as np

from .baselines import BASELINES
from .config import load_config

log = logging.getLogger("trader")
ALL_MODELS = BASELINES + ["transformer"]


def cmd_fetch(cfg, args):
    from .data import fetch_prices, prices_path

    long = fetch_prices(cfg)
    path = prices_path(cfg.data.dir)
    dates = long.index.get_level_values("date")
    log.info("saved %d %s bars for %d symbols (%s to %s) -> %s", len(long), cfg.data.interval,
             long.index.get_level_values(1).nunique(), dates.min(), dates.max(), path)


def _features(cfg):
    from .data import load_panel
    from .features import features_from_config

    panel = load_panel(cfg)
    return panel, features_from_config(panel, cfg)


def cmd_analyze(cfg, args):
    from .analysis import run_analysis
    from .data import load_panel

    path = run_analysis(load_panel(cfg), cfg)
    log.info("analysis report -> %s", path)


def cmd_walkforward(cfg, args):
    from .walkforward import save_predictions, walk_forward

    _, fs = _features(cfg)
    for name in args.models:
        t0 = time.time()
        df = walk_forward(fs, name, cfg)
        path = save_predictions(df, cfg, name)
        log.info("%s: %d out-of-sample predictions in %.0fs -> %s", name, len(df), time.time() - t0, path)


def cmd_evaluate(cfg, args):
    from .evaluate import run_evaluation
    from .walkforward import load_all_predictions

    panel, fs = _features(cfg)
    preds = load_all_predictions(cfg)
    if not preds:
        raise SystemExit("no predictions found - run `python -m trader walkforward` first")
    path = run_evaluation(preds, fs, panel, cfg)
    log.info("evaluation report -> %s", path)


def cmd_train(cfg, args):
    """Fit the production transformer on all available history."""
    from .live import model_path
    from .transformer import TransformerPredictor

    _, fs = _features(cfg)
    t_idx, n_idx = fs.samples()
    model = TransformerPredictor(cfg.model, cfg.training).fit(fs, t_idx, n_idx)
    last = fs.dates[t_idx.max()]
    model.save(model_path(cfg), meta={"trained_through": str(last), "horizon": fs.horizon,
                                      "execution": fs.execution, "interval": cfg.data.interval,
                                      "n_samples": int(len(t_idx))})
    log.info("trained on %d samples through %s -> %s", len(t_idx), last, model_path(cfg))


def cmd_trade(cfg, args):
    from .live import run_trade

    run_trade(cfg, execute=args.execute)


def cmd_monitor(cfg, args):
    from .live import run_monitor

    run_monitor(cfg)


def cmd_gui(cfg, args):
    import subprocess
    import sys
    from pathlib import Path

    app = Path(__file__).with_name("app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true",
                    "--server.port", str(args.port), "--", "--config", args.config], check=False)


def cmd_research(cfg, args):
    """Full research pipeline: data -> analysis -> walk-forward for all models -> evaluation."""
    cmd_fetch(cfg, args)
    cmd_analyze(cfg, args)
    cmd_walkforward(cfg, args)
    cmd_evaluate(cfg, args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="trader", description="Quantitative tech-stock research and trading.")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download daily prices into the data cache")
    sub.add_parser("analyze", help="statistical analysis report of returns/volatility/correlation")
    for name, helptext in (("walkforward", "walk-forward out-of-sample predictions"),
                           ("research", "fetch + analyze + walkforward + evaluate")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    sub.add_parser("evaluate", help="compare models and backtest the strategy after costs")
    sub.add_parser("train", help="train the production transformer on all data")
    tp = sub.add_parser("trade", help="compute today's orders and (optionally) send them to IB")
    tp.add_argument("--execute", action="store_true", help="actually submit orders (default: dry run)")
    sub.add_parser("monitor", help="score live predictions against realised returns")
    gp = sub.add_parser("gui", help="interactive analysis and simulation app in the browser")
    gp.add_argument("--port", type=int, default=8501)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    cfg.paths.ensure()
    np.seterr(all="ignore")
    globals()[f"cmd_{args.command}"](cfg, args)


if __name__ == "__main__":
    main()
