"""Generate candidate next-experiment hypotheses from LAB_JOURNAL.md via the Claude API (Opus by
default) -- the "ideation" step that's still manual in the propose -> dispatch -> auto-decide ->
auto-log loop (campaign.py handles dispatch/decide/log; this is the missing "what should we try
next" piece). Deliberately NOT wired to auto-dispatch anything -- this only prints/returns text
hypotheses for a human (or a separate, explicit follow-up step) to review and decide whether to
turn into actual configs. Keeping ideation and execution as separate steps means an API call can
never itself spend RunPod GPU money -- see LAB_JOURNAL.md's 2026-07-21 discussion of why a fully
autonomous "propose AND dispatch" loop needs its own, more careful design (cost caps, candidate
whitelisting) before being built.

Requires ANTHROPIC_API_KEY (see infra/set_secret.sh -- run it yourself in a terminal, never via
the assistant, then add ANTHROPIC_API_KEY to infra/load_secrets.sh's secret list if not already
there).

Usage:
    source infra/load_secrets.sh
    python -m infra.hypothesize --journal LAB_JOURNAL.md --n 5
"""

import argparse
import logging
import os
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are helping plan the next experiments for a graph neural network + \
diffusion-augmentation fraud detection research project. Below is the FULL lab journal: every \
experiment run so far, in chronological order, including what worked, what didn't, and why.

Your job: propose {n} concrete, DIFFERENT candidate next experiments. For each one:
1. A short name/title.
2. The specific hypothesis being tested (what result would confirm or refute it).
3. Why it's plausible GIVEN what's already been tried and ruled out in the journal -- do not \
propose something that's already been tested and found negative, unless you have a specific \
reason the prior test was confounded (say what that reason is).
4. Roughly how it would be implemented (which files/config fields would change).
5. What would make this NOT worth pursuing (a concrete falsification condition), so a quick \
single-seed or small-scale check can rule it out cheaply before committing to a full multi-seed run.

Prioritize ideas that are genuinely informative given what's already been ruled out -- not \
generic GNN tricks that haven't been contextualized against this project's specific findings \
(e.g. RF beating every GNN variant on Elliptic, diffusion not stacking additively with anything, \
the "hard core" of feature-space-outlier fraud neither RF nor GNN catches).

LAB JOURNAL:
{journal}
"""


def generate_hypotheses(journal_path: str, n: int = 5, model: str = "claude-opus-4-8",
                         max_tokens: int = 4000) -> str:
    journal_text = (ROOT / journal_path).read_text()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(n=n, journal=journal_text),
        }],
    )
    return message.content[0].text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="LAB_JOURNAL.md")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args()

    hypotheses = generate_hypotheses(args.journal, args.n, args.model)
    print(hypotheses)


if __name__ == "__main__":
    main()
