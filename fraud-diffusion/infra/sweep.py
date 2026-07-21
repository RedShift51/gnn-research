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
import copy
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import runpod
import yaml

from infra.runpod_serverless import invoke

logger = logging.getLogger(__name__)


def _default_run_name(base_config_path: str, point: dict) -> str:
    stem = base_config_path.split("/")[-1].removesuffix(".yaml")
    tag = "_".join(f"{dotted.split('.')[-1]}{value}" for dotted, value in point.items())
    return f"{stem}_sweep_{tag}"


def _set_nested(config: dict, dotted_path: str, value) -> None:
    """Set config[a][b][c]=value for dotted_path="a.b.c", at ANY depth (not just one level) --
    creates intermediate dicts as needed. Needed for configs like diffusion.spectral.lambda_spectral
    (two levels deep) that sweep_config()'s section-keyed **overrides interface can't reach."""
    keys = dotted_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_grid(base_config_path: str, param_grid: dict, run_name_fn=None) -> list[tuple[str, dict, dict]]:
    """param_grid: {"loss.alpha": [0.25, 0.5], "diffusion.spectral.lambda_spectral": [0.01, 0.1]}
    (dotted keys at any depth) -> the full cartesian product, one config per combination. Returns
    [(run_name, config_dict, point_dict), ...]."""
    run_name_fn = run_name_fn or (lambda point: _default_run_name(base_config_path, point))
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    keys = list(param_grid.keys())
    jobs = []
    for combo in itertools.product(*param_grid.values()):
        point = dict(zip(keys, combo))
        config = copy.deepcopy(base_config)
        for dotted_key, value in point.items():
            _set_nested(config, dotted_key, value)
        run_name = run_name_fn(point)
        config.setdefault("journal", {})["run_name"] = run_name
        jobs.append((run_name, config, point))
    return jobs


def _invoke_with_retry(endpoint_id, job_input, timeout, run_name, max_attempts=3):
    """Some large full-batch configs (PaySim's full graph, hidden_dim=128) sit right at a 24GB
    GPU's memory ceiling and OOM intermittently rather than deterministically — directly observed
    the SAME config succeed and fail across repeated attempts, not tied to any code difference.
    Retrying lands a clean result most of the time without needing to shrink model capacity or
    rearchitect full-batch training into mini-batch (PaySim's own established fix for GAT) just to
    chase an intermittent failure. Each retry gets a fresh worker via RunPod's own scheduling, and
    the worker-kill-on-OOM fix (serverless/handler.py) prevents a poisoned worker from being handed
    the retry."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return invoke(endpoint_id, job_input, timeout)
        except Exception as e:
            last_error = e
            logger.warning(f"[{run_name}] attempt {attempt}/{max_attempts} failed: {e}")
    raise last_error


def run_sweep(endpoint_id: str, base_config_path: str, param_grid: dict,
              max_workers: int = 4, timeout: int = 3600, preprocess: bool = True,
              max_attempts: int = 3) -> dict:
    jobs = build_grid(base_config_path, param_grid)
    logger.info(f"Dispatching {len(jobs)} jobs ({', '.join(param_grid.keys())} grid) to endpoint {endpoint_id}...")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(_invoke_with_retry, endpoint_id, {"config_dict": config, "preprocess": preprocess},
                        timeout, run_name, max_attempts):
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
                logger.error(f"[{run_name}] FAILED after {max_attempts} attempts — point={point}: {e}")
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
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    param_grid = json.loads(args.grid)
    results = run_sweep(
        args.endpoint_id, args.base_config, param_grid,
        max_workers=args.max_workers, timeout=args.timeout, preprocess=not args.no_preprocess,
        max_attempts=args.max_attempts,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
