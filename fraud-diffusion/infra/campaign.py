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
import re
from datetime import date
from pathlib import Path

import runpod
import yaml

from infra.multi_seed import compare, format_report

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _run_name(config_path: str) -> str:
    """journal.run_name if the config declares one, else the bare filename -- just a label, not
    used to locate anything."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("journal", {}).get("run_name", config_path)
    except Exception:
        return config_path


def append_campaign_journal_entry(journal_path: str, best_config_path: str, candidate_path: str,
                                   report: str, decision: str, promoted: bool) -> int:
    """Auto-appends a Run entry to LAB_JOURNAL.md for one campaign comparison -- same run-numbering
    convention as training/train_gnn.py's append_journal_entry() (count existing '\\n## [' entries
    + 1), so campaign runs interleave correctly with training-run entries in the same sequential
    numbering. Observations/Next are left as manual fill-in placeholders, same as train_gnn.py's own
    auto-entries: a script can report WHAT happened (the numbers, the decision) but not WHY, or
    what it means for the project -- that's still a human/model judgment call, not automated."""
    path = ROOT / journal_path
    # max(existing "Run N") + 1, NOT a raw header count -- a plain count collides whenever the
    # file already has any gap or duplicate (confirmed: this journal already had a few pre-existing
    # collisions from before this session, e.g. Run 24/26/44/58, and this exact function produced
    # one more, a duplicate Run 65, when a manual edit and a campaign run landed close together).
    # Max-based numbering is robust to that history regardless of how it got there.
    text = path.read_text() if path.exists() else ""
    existing_run_numbers = [int(m) for m in re.findall(r"\n## \[.*?\] Run (\d+)", text)]
    run_id = max(existing_run_numbers, default=0) + 1

    label_a, label_b = _run_name(best_config_path), _run_name(candidate_path)
    entry = f"""
## [{date.today().isoformat()}] Run {run_id} — campaign: {label_b} vs {label_a}
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best={best_config_path},
  candidate={candidate_path}).
{report}
- Decision: {decision}
- Observations: (fill in manually)
- Next: (fill in manually)
"""
    with open(path, "a") as f:
        f.write(entry)
    logger.info(f"Appended campaign run {run_id} to {journal_path}")
    return run_id


def run_campaign(endpoint_id: str, best_config_path: str, candidate_paths: list, seeds: list,
                  metric: str = "f1_macro", split: str = "test", max_workers: int = 4,
                  timeout: int = 3600, preprocess: bool = True, significance: float = 0.05,
                  journal: bool = True) -> dict:
    current_best = best_config_path
    history = []

    for candidate_path in candidate_paths:
        logger.info(f"Campaign: comparing current best ({current_best}) vs candidate ({candidate_path})...")
        try:
            result = compare(endpoint_id, current_best, candidate_path, seeds, metric, split,
                              max_workers, timeout, preprocess)
            report = format_report(result, current_best, candidate_path, metric)
        except Exception as e:
            # A crash on ONE candidate (confirmed: a network drop mid-poll took down the entire
            # remaining queue, including candidates that had nothing to do with the failure) must
            # not stop the rest of the queue -- record it and move on to the next candidate against
            # the SAME current_best (nothing was promoted, so this is the correct fallback state).
            logger.error(f"Campaign: {candidate_path} crashed ({e}) -- skipping, keeping current best")
            history.append({
                "best_before": current_best, "candidate": candidate_path, "report": None,
                "promoted": False, "decision": f"CRASHED: {e}", "journal_run_id": None,
            })
            continue

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

        run_id = None
        if journal:
            # Use the candidate config's OWN declared journal path (every config in this repo
            # sets journal.path, normally LAB_JOURNAL.md) rather than hard-coding it, so a
            # campaign over configs pointed at a different journal file still lands correctly.
            journal_path = None
            try:
                with open(candidate_path) as f:
                    journal_path = yaml.safe_load(f).get("journal", {}).get("path")
            except Exception:
                pass
            if journal_path:
                run_id = append_campaign_journal_entry(
                    journal_path, current_best, candidate_path, report, decision, promoted,
                )

        history.append({
            "best_before": current_best,
            "candidate": candidate_path,
            "report": report,
            "promoted": promoted,
            "decision": decision,
            "journal_run_id": run_id,
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
    run_p.add_argument("--no-journal", action="store_true", help="skip auto-appending to LAB_JOURNAL.md")

    args = parser.parse_args()
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    seeds = list(range(args.n_seeds))

    if args.command == "run":
        result = run_campaign(args.endpoint_id, args.best, args.candidates, seeds, args.metric,
                               args.split, args.max_workers, args.timeout, not args.no_preprocess,
                               args.significance, journal=not args.no_journal)
        print(f"\nFinal best: {result['final_best']}\n")
        for entry in result["history"]:
            print(f"--- {entry['candidate']} ---")
            print(entry["report"])
            print(entry["decision"])
            if entry["journal_run_id"]:
                print(f"(logged as Run {entry['journal_run_id']} in LAB_JOURNAL.md)")
            print()


if __name__ == "__main__":
    main()
