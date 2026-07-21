"""Grid-dispatch sweep runner — the first step toward automated experiment loops.

Replaces what had been done by hand for every sweep so far (write one YAML config per
hyperparameter point, commit, push, fire one `invoke` call per point via the CLI — see
LAB_JOURNAL.md's alpha sweeps): define a parameter grid once, generate one config_dict per
combination via runpod_serverless.sweep_config(), and dispatch them all to RunPod IN PARALLEL
(threaded — invoke() blocks on a poll loop, so threads overlap the wait). Every job still logs to
wandb server-side (train_gnn.py's init_wandb), so the wandb UI gives the same side-by-side
comparison as the printed summary table here.

This is deliberately a fixed grid, not adaptive search — a real wandb Sweep (bayesian/hyperband,
picking the next point from prior results) is the natural next step once this pattern is proven,
but needs an adapter since our training runs on RunPod, not in-process (see infra/wandb_sweep.py).

Usage:
    python -m infra.sweep --endpoint-id YOUR_ENDPOINT_ID --base-config configs/elliptic_diffusion_alpha025.yaml \
        --grid '{"loss.alpha": [0.25, 0.4, 0.5], "augment.clamp_std": [2, 3, 4]}'
"""

import argparse
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import runpod

from infra.runpod_serverless import invoke, sweep_config

logger = logging.getLogger(__name__)


def _default_run_name(base_config_path: str, point: dict) -> str:
    stem = base_config_path.split("/")[-1].removesuffix(".yaml")
    tag = "_".join(f"{dotted.split('.')[-1]}{value}" for dotted, value in point.items())
    return f"{stem}_sweep_{tag}"


def build_grid(base_config_path: str, param_grid: dict, run_name_fn=None) -> list[tuple[str, dict, dict]]:
    """param_grid: {"loss.alpha": [0.25, 0.5], "augment.clamp_std": [2, 3]} (dotted "section.field"
    keys, matching sweep_config()'s section-keyed overrides) -> the full cartesian product, one
    config per combination. Returns [(run_name, config_dict, point_dict), ...]."""
    run_name_fn = run_name_fn or (lambda point: _default_run_name(base_config_path, point))
    keys = list(param_grid.keys())
    jobs = []
    for combo in itertools.product(*param_grid.values()):
        point = dict(zip(keys, combo))
        overrides = {}
        for dotted_key, value in point.items():
            section, field = dotted_key.split(".", 1)
            overrides.setdefault(section, {})[field] = value
        run_name = run_name_fn(point)
        config = sweep_config(base_config_path, run_name, **overrides)
        jobs.append((run_name, config, point))
    return jobs


def run_sweep(endpoint_id: str, base_config_path: str, param_grid: dict,
              max_workers: int = 4, timeout: int = 3600, preprocess: bool = True) -> dict:
    jobs = build_grid(base_config_path, param_grid)
    logger.info(f"Dispatching {len(jobs)} jobs ({', '.join(param_grid.keys())} grid) to endpoint {endpoint_id}...")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(invoke, endpoint_id, {"config_dict": config, "preprocess": preprocess}, timeout):
                (run_name, point)
            for run_name, config, point in jobs
        }
        for future in as_completed(future_to_job):
            run_name, point = future_to_job[future]
            try:
                result = future.result()
                results[run_name] = {"point": point, "output": result.get("output", result)}
                test = results[run_name]["output"].get("test", {})
                logger.info(f"[{run_name}] done — point={point} "
                            f"test_f1={test.get('f1_macro')} test_auc={test.get('auc_roc')}")
            except Exception as e:
                results[run_name] = {"point": point, "error": str(e)}
                logger.error(f"[{run_name}] FAILED — point={point}: {e}")
    return results


def print_summary(results: dict) -> None:
    rows = []
    for run_name, r in results.items():
        test = r.get("output", {}).get("test", {})
        rows.append((run_name, r["point"], test.get("f1_macro"), test.get("auc_roc"), test.get("g_mean")))
    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))

    print("\n| run | point | test_f1 | test_auc | test_g_mean |")
    print("|---|---|---|---|---|")
    for run_name, point, f1, auc, g in rows:
        f1_s = f"{f1:.4f}" if f1 is not None else "FAILED"
        auc_s = f"{auc:.4f}" if auc is not None else "-"
        g_s = f"{g:.4f}" if g is not None else "-"
        print(f"| {run_name} | {point} | {f1_s} | {auc_s} | {g_s} |")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--grid", required=True, help='JSON dict, e.g. \'{"loss.alpha": [0.25, 0.5]}\'')
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    param_grid = json.loads(args.grid)
    results = run_sweep(
        args.endpoint_id, args.base_config, param_grid,
        max_workers=args.max_workers, timeout=args.timeout, preprocess=not args.no_preprocess,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
