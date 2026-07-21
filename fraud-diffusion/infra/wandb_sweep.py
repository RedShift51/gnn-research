"""wandb Sweep adapter -- the second step toward automated experiment loops (see infra/sweep.py's
docstring for the first, fixed-grid step). A real wandb Sweep controller (bayesian/hyperband)
picks the NEXT hyperparameter point from prior trial results, instead of a fixed grid -- useful
once there are more dimensions to search than fit in a manual grid, or you want smarter-than-grid
search.

Training itself still runs on RunPod, not locally -- wandb's normal `wandb agent` mechanism spawns
your training function as a local subprocess and reads its sweep-chosen hyperparameters via
wandb.config, so this script bridges that: the sweep AGENT runs locally (one cheap process on your
Mac), and for each trial it dispatches ONE RunPod job with a config_dict built from wandb's
sweep-chosen hyperparameters (dotted "section.field" parameter names, same convention as
infra/sweep.py), then reports the final test/val metrics back to wandb so the controller can plan
the next point.

The remote job's own wandb logging is disabled per-trial (config["wandb"]["enabled"]=False) --
otherwise every trial would create a second, orphaned wandb run inside the ephemeral RunPod
container with no way to attach it to the sweep. The local agent's run IS the sweep trial's run;
it only gets the final val/test metrics (not per-epoch curves), since that's all it has after
invoke() returns. Forwarding per-epoch curves too would mean threading the sweep run's ID into the
remote job's env so its own wandb.init() resumes that same run -- a reasonable next step, not done
here yet.

Usage:
    python -m infra.wandb_sweep create --base-config configs/elliptic_diffusion_alpha025.yaml \\
        --method bayes --metric test_f1_macro --goal maximize \\
        --params '{"loss.alpha": {"min": 0.25, "max": 0.95}, "augment.clamp_std": {"values": [2, 3, 4, 5]}}'
    # prints a sweep_id, then:
    python -m infra.wandb_sweep agent SWEEP_ID --endpoint-id YOUR_ENDPOINT_ID \\
        --base-config configs/elliptic_diffusion_alpha025.yaml --count 10
"""

import argparse
import copy
import json
import logging
import os

import runpod
import wandb
import yaml

from infra.runpod_serverless import invoke

logger = logging.getLogger(__name__)


def _apply_point(base_config_path: str, point: dict) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    for dotted_key, value in point.items():
        section, field = dotted_key.split(".", 1)
        config.setdefault(section, {})[field] = value
    config.setdefault("wandb", {})["enabled"] = False
    return config


def create_sweep(project: str, method: str, metric: str, goal: str, params: dict) -> str:
    sweep_definition = {
        "method": method,
        "metric": {"name": metric, "goal": goal},
        "parameters": params,
    }
    sweep_id = wandb.sweep(sweep_definition, project=project)
    logger.info(f"Created sweep {sweep_id} (project={project}, method={method}, metric={metric}/{goal})")
    return sweep_id


def run_agent(sweep_id: str, project: str, endpoint_id: str, base_config_path: str,
              count: int, timeout: int = 3600) -> None:
    def train_one():
        run = wandb.init()
        point = dict(run.config)
        config = _apply_point(base_config_path, point)
        run.name = "sweep_" + "_".join(f"{k.split('.')[-1]}{v}" for k, v in point.items())
        logger.info(f"[{run.name}] dispatching to RunPod with point={point}")

        result = invoke(endpoint_id, {"config_dict": config, "preprocess": True}, timeout)
        output = result.get("output", result)
        test_metrics = {k: v for k, v in output.get("test", {}).items() if not isinstance(v, dict)}
        val_metrics = {k: v for k, v in output.get("val", {}).items() if not isinstance(v, dict)}

        run.log({f"test_{k}": v for k, v in test_metrics.items()})
        run.log({f"val_{k}": v for k, v in val_metrics.items()})
        for k, v in test_metrics.items():
            run.summary[f"test_{k}"] = v
        logger.info(f"[{run.name}] done — test={test_metrics}, val={val_metrics}")

    wandb.agent(sweep_id, function=train_one, project=project, count=count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    create_p = sub.add_parser("create")
    create_p.add_argument("--project", default="fraud-diffusion")
    create_p.add_argument("--method", default="bayes", choices=["grid", "random", "bayes"])
    create_p.add_argument("--metric", default="test_f1_macro")
    create_p.add_argument("--goal", default="maximize", choices=["maximize", "minimize"])
    create_p.add_argument("--params", required=True,
                           help='JSON dict of wandb sweep parameter specs, e.g. '
                                '\'{"loss.alpha": {"min": 0.25, "max": 0.95}}\'')

    agent_p = sub.add_parser("agent")
    agent_p.add_argument("sweep_id")
    agent_p.add_argument("--project", default="fraud-diffusion")
    agent_p.add_argument("--endpoint-id", required=True)
    agent_p.add_argument("--base-config", required=True)
    agent_p.add_argument("--count", type=int, default=10)
    agent_p.add_argument("--timeout", type=int, default=3600)

    args = parser.parse_args()
    if args.command == "create":
        params = json.loads(args.params)
        create_sweep(args.project, args.method, args.metric, args.goal, params)
    elif args.command == "agent":
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        run_agent(args.sweep_id, args.project, args.endpoint_id, args.base_config, args.count, args.timeout)


if __name__ == "__main__":
    main()
