"""Sequential experiment campaign runner -- one step closer to a "loop" (automated eval / adaptive
search) than infra/multi_seed.py's compare()/format_report() alone. Takes a queue of candidate
configs, multi-seed-compares each against the CURRENT BEST config in turn, and auto-promotes a
candidate to "current best" only if it wins with both a better mean AND a statistically
significant (p < significance) paired Wilcoxon test -- otherwise the existing best is kept and the
next candidate is tried against it unchanged. Removes the manual "run compare, read the numbers,
decide, hand-write a journal entry" cycle for testing a queue of candidates one after another.

Deliberately NOT a Bayesian/RL search over a continuous config space -- that was discussed and
deprioritized in favor of this simpler wrapper-first approach (2026-07-21). This only sequences
and auto-decides across a small, human-curated list of candidate configs.

Usage:
    python -m infra.campaign run --endpoint-id YOUR_ENDPOINT_ID \\
        --best configs/elliptic_full_graphsage_diff.yaml \\
        --candidates configs/elliptic_graphsage_diff_encoder.yaml configs/elliptic_full_graphsage_gated.yaml \\
        --n-seeds 5 --metric f1_macro
"""

import argparse
import logging
import os

import runpod

from infra.multi_seed import compare, format_report

logger = logging.getLogger(__name__)


def run_campaign(endpoint_id: str, best_config_path: str, candidate_paths: list, seeds: list,
                  metric: str = "f1_macro", split: str = "test", max_workers: int = 4,
                  timeout: int = 3600, preprocess: bool = True, significance: float = 0.05) -> dict:
    current_best = best_config_path
    history = []

    for candidate_path in candidate_paths:
        logger.info(f"Campaign: comparing current best ({current_best}) vs candidate ({candidate_path})...")
        result = compare(endpoint_id, current_best, candidate_path, seeds, metric, split,
                          max_workers, timeout, preprocess)
        report = format_report(result, current_best, candidate_path, metric)

        mean_best = result["config_a"]["summary"]["mean"]
        mean_candidate = result["config_b"]["summary"]["mean"]
        wilcoxon = result["wilcoxon"] or {}
        p_value = wilcoxon.get("p_value")

        promoted = (
            mean_best is not None and mean_candidate is not None
            and mean_candidate > mean_best
            and p_value is not None and p_value < significance
        )

        decision = (
            f"PROMOTED to new best (mean {mean_candidate:.4f} > {mean_best:.4f}, p={p_value:.4f} < {significance})"
            if promoted else
            f"kept existing best (candidate did not win significantly: "
            f"mean {mean_candidate if mean_candidate is not None else 'N/A'} vs "
            f"{mean_best if mean_best is not None else 'N/A'}, "
            f"p={p_value if p_value is not None else 'N/A'})"
        )
        logger.info(f"Campaign decision for {candidate_path}: {decision}")

        history.append({
            "best_before": current_best,
            "candidate": candidate_path,
            "report": report,
            "promoted": promoted,
            "decision": decision,
        })

        if promoted:
            current_best = candidate_path

    return {"final_best": current_best, "history": history}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--endpoint-id", required=True)
    run_p.add_argument("--best", required=True)
    run_p.add_argument("--candidates", nargs="+", required=True)
    run_p.add_argument("--n-seeds", type=int, default=5)
    run_p.add_argument("--metric", default="f1_macro")
    run_p.add_argument("--split", default="test")
    run_p.add_argument("--max-workers", type=int, default=4)
    run_p.add_argument("--timeout", type=int, default=3600)
    run_p.add_argument("--no-preprocess", action="store_true")
    run_p.add_argument("--significance", type=float, default=0.05)

    args = parser.parse_args()
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    seeds = list(range(args.n_seeds))

    if args.command == "run":
        result = run_campaign(args.endpoint_id, args.best, args.candidates, seeds, args.metric,
                               args.split, args.max_workers, args.timeout, not args.no_preprocess,
                               args.significance)
        print(f"\nFinal best: {result['final_best']}\n")
        for entry in result["history"]:
            print(f"--- {entry['candidate']} ---")
            print(entry["report"])
            print(entry["decision"])
            print()


if __name__ == "__main__":
    main()
