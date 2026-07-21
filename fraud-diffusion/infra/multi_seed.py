"""Multi-seed runner for statistical significance -- CLAUDE.md's own success criteria calls for
"10-seed statistical significance" and a Wilcoxon signed-rank test, and every result logged so far
in LAB_JOURNAL.md is a single seed (42). Dispatches the SAME config N times with different seeds
(RunPod jobs, parallel like infra/sweep.py), reports mean+-std per metric, and optionally a PAIRED
Wilcoxon signed-rank test against a second config's per-seed results (e.g. baseline vs diffusion) --
paired because the same seed list drives both configs, controlling for run-to-run variance instead
of treating the two sets of runs as independent samples.

Usage:
    python -m infra.multi_seed run --endpoint-id YOUR_ENDPOINT_ID \\
        --config configs/elliptic_full.yaml --n-seeds 10
    python -m infra.multi_seed compare --endpoint-id YOUR_ENDPOINT_ID \\
        --config-a configs/elliptic_full.yaml --config-b configs/elliptic_diffusion_alpha025.yaml \\
        --n-seeds 10 --metric f1_macro
"""

import argparse
import copy
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import runpod
import yaml
from scipy import stats

from infra.runpod_serverless import invoke

logger = logging.getLogger(__name__)


def _with_seed(base_config_path: str, seed: int, run_name_suffix: str) -> dict:
    """seed is a top-level config key (not nested under a section), so it needs its own override
    helper rather than reusing infra/sweep.py's build_grid()/sweep_config(), which only knows how
    to override "section.field"-style nested keys."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["seed"] = seed
    base_run_name = config.get("journal", {}).get("run_name", run_name_suffix)
    config.setdefault("journal", {})["run_name"] = f"{base_run_name}_seed{seed}"
    return config


def run_multi_seed(endpoint_id: str, config_path: str, seeds: list, max_workers: int = 4,
                    timeout: int = 3600, preprocess: bool = True) -> dict:
    logger.info(f"Dispatching {len(seeds)} seeds ({seeds}) of {config_path} to endpoint {endpoint_id}...")
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_seed = {
            pool.submit(invoke, endpoint_id,
                        {"config_dict": _with_seed(config_path, seed, config_path), "preprocess": preprocess},
                        timeout): seed
            for seed in seeds
        }
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                output = result.get("output", result)
                results[seed] = output
                test = output.get("test", {})
                logger.info(f"[seed={seed}] done — test_f1={test.get('f1_macro')} test_auc={test.get('auc_roc')}")
            except Exception as e:
                results[seed] = {"error": str(e)}
                logger.error(f"[seed={seed}] FAILED: {e}")
    return results


def summarize(results: dict, split: str = "test", metric: str = "f1_macro") -> dict:
    values = [r[split][metric] for r in results.values() if split in r]
    if not values:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def compare(endpoint_id: str, config_a_path: str, config_b_path: str, seeds: list,
            metric: str = "f1_macro", split: str = "test", max_workers: int = 4,
            timeout: int = 3600, preprocess: bool = True) -> dict:
    """Runs both configs across the SAME seed list, then a paired Wilcoxon signed-rank test on
    the per-seed metric values. Paired, not independent-samples, because seed i of config A and
    seed i of config B share initialization/data-shuffling noise — the paired test isolates the
    configs' difference from that shared noise, which an unpaired test (e.g. Mann-Whitney) would not."""
    results_a = run_multi_seed(endpoint_id, config_a_path, seeds, max_workers, timeout, preprocess)
    results_b = run_multi_seed(endpoint_id, config_b_path, seeds, max_workers, timeout, preprocess)

    paired_a, paired_b = [], []
    for seed in seeds:
        ra, rb = results_a.get(seed, {}), results_b.get(seed, {})
        if split in ra and split in rb:
            paired_a.append(ra[split][metric])
            paired_b.append(rb[split][metric])

    summary_a = summarize(results_a, split, metric)
    summary_b = summarize(results_b, split, metric)

    wilcoxon_result = None
    if len(paired_a) >= 2:
        try:
            stat, p_value = stats.wilcoxon(paired_a, paired_b)
            wilcoxon_result = {"statistic": float(stat), "p_value": float(p_value), "n_pairs": len(paired_a)}
        except ValueError as e:
            # e.g. all differences are zero, or too few non-zero pairs -- wilcoxon can't run
            wilcoxon_result = {"error": str(e), "n_pairs": len(paired_a)}

    return {
        "config_a": {"path": config_a_path, "summary": summary_a, "raw": results_a},
        "config_b": {"path": config_b_path, "summary": summary_b, "raw": results_b},
        "wilcoxon": wilcoxon_result,
    }


def format_report(result: dict, label_a: str, label_b: str, metric: str = "f1_macro") -> str:
    """Ready-to-paste markdown for a LAB_JOURNAL.md run entry -- removes the "hand-build the
    table each time" friction that kept turning "test a candidate" into a two-step manual dance
    (quick single-seed check, then remember to multi-seed it properly). Call this immediately
    after compare() instead of hand-formatting the summary/Wilcoxon dict."""
    sa, sb = result["config_a"]["summary"], result["config_b"]["summary"]
    w = result["wilcoxon"] or {}
    lines = [
        f"| Config | Mean {metric} | Std | n |",
        "|---|---|---|---|",
        f"| {label_a} | {sa['mean']:.4f} | {sa['std']:.4f} | {sa['n']} |",
        f"| {label_b} | {sb['mean']:.4f} | {sb['std']:.4f} | {sb['n']} |",
        "",
    ]
    if "p_value" in w:
        lines.append(f"Wilcoxon signed-rank: statistic={w['statistic']:.2f}, "
                      f"p={w['p_value']:.4f}, n_pairs={w['n_pairs']}")
    elif w:
        lines.append(f"Wilcoxon signed-rank: {w}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--endpoint-id", required=True)
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--n-seeds", type=int, default=10)
    run_p.add_argument("--max-workers", type=int, default=4)
    run_p.add_argument("--timeout", type=int, default=3600)
    run_p.add_argument("--no-preprocess", action="store_true")

    compare_p = sub.add_parser("compare")
    compare_p.add_argument("--endpoint-id", required=True)
    compare_p.add_argument("--config-a", required=True)
    compare_p.add_argument("--config-b", required=True)
    compare_p.add_argument("--n-seeds", type=int, default=10)
    compare_p.add_argument("--metric", default="f1_macro")
    compare_p.add_argument("--split", default="test")
    compare_p.add_argument("--max-workers", type=int, default=4)
    compare_p.add_argument("--timeout", type=int, default=3600)
    compare_p.add_argument("--no-preprocess", action="store_true")

    args = parser.parse_args()
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    seeds = list(range(args.n_seeds))

    if args.command == "run":
        results = run_multi_seed(args.endpoint_id, args.config, seeds, args.max_workers,
                                  args.timeout, not args.no_preprocess)
        print(summarize(results))
    elif args.command == "compare":
        result = compare(args.endpoint_id, args.config_a, args.config_b, seeds,
                          args.metric, args.split, args.max_workers, args.timeout, not args.no_preprocess)
        print(format_report(result, args.config_a, args.config_b, args.metric))


if __name__ == "__main__":
    main()
