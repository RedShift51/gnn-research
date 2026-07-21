"""Serverless job management for RunPod — the analogue of a Modal app.py.

Modal collapses "define function -> build image -> deploy -> invoke" into one decorated
Python function you call with `.remote()`. RunPod Serverless splits those steps explicitly:

    build_and_push_image()  -- docker build + push serverless/Dockerfile to GHCR
    deploy()                -- create/update a RunPod Template + Endpoint pointing at that image
    invoke(job_input)        -- call the endpoint, wait for the result (like Modal's .remote())

serverless/handler.py is the code that actually runs inside the container per job — the
equivalent of the body of a Modal @app.function.

Usage (run from the fraud-diffusion/ project root, after `source infra/load_secrets.sh`):
    python -m infra.runpod_serverless build
    python -m infra.runpod_serverless deploy
    python -m infra.runpod_serverless invoke --config configs/paysim.yaml
"""

import argparse
import copy
import logging
import os
import subprocess
import time

import runpod
import yaml

logger = logging.getLogger(__name__)

IMAGE_NAME = "fraud-diffusion-serverless"
TEMPLATE_NAME = "fraud-diffusion-serverless"
ENDPOINT_NAME = "fraud-diffusion-serverless"


def sweep_config(base_config_path: str, run_name: str, **overrides) -> dict:
    """Load a base config (e.g. configs/paysim_full_gat.yaml) and apply nested-dict overrides,
    for one-off hyperparameter sweeps via invoke()'s config_dict — no new YAML file, no image
    rebuild. E.g.:
        sweep_config("configs/paysim_full_gat.yaml", "gat_oversample_0.05",
                     train={"oversample_fraud_frac": 0.05})
    """
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    config.setdefault("journal", {})["run_name"] = run_name
    return config


def _image_ref(tag: str = "latest") -> str:
    github_username = os.environ["GITHUB_USERNAME"]
    return f"ghcr.io/{github_username}/{IMAGE_NAME}:{tag}"


def build_and_push_image(tag: str = "latest") -> str:
    """docker build + push to GHCR. Requires GITHUB_USERNAME + GHCR_TOKEN loaded (see
    infra/load_secrets.sh) and Docker running locally."""
    image_ref = _image_ref(tag)
    github_username = os.environ["GITHUB_USERNAME"]
    ghcr_token = os.environ["GHCR_TOKEN"]

    subprocess.run(
        ["docker", "login", "ghcr.io", "-u", github_username, "--password-stdin"],
        input=ghcr_token, text=True, check=True,
    )
    subprocess.run(
        ["docker", "build", "-f", "serverless/Dockerfile", "-t", image_ref, "."],
        check=True,
    )
    subprocess.run(["docker", "push", image_ref], check=True)
    logger.info(f"Pushed {image_ref}")
    return image_ref


def deploy(tag: str = "latest", gpu_ids: str = "ADA_24") -> str:
    """Create (or recreate) the RunPod Template + Serverless Endpoint for our image.

    gpu_ids is a comma-separated list of RunPod GPU *pool* IDs, not a literal GPU name — e.g.
    "ADA_24" (RTX 4090 class, 24GB Ada Lovelace), not "NVIDIA GeForce RTX 4090". See
    https://docs.runpod.io/references/gpu-types#gpu-pools for the full list.

    "ADA_48_PRO" (48GB tier) is NOT Ada-only despite the name — it can schedule onto Blackwell
    RTX PRO 6000 MIG slices (sm_120), which our image's PyTorch (CUDA 12.4, built for up to
    sm_90) can't run on at all ("no kernel image is available for execution on the device").
    Exclude incompatible cards with "-<GPU name>", e.g.
    "ADA_48_PRO,-NVIDIA RTX PRO 6000 Blackwell Server Edition".

    The GHCR package is private (inherited from the private GitHub repo — confirmed via
    the GitHub API: anonymous pull gets 403), so RunPod needs registry credentials to pull
    it. We register those via create_container_registry_auth and reference the returned id
    in the template, rather than relying on public pull.

    Kaggle credentials: pass them as plain env here (pulled from local Keychain via
    load_secrets.sh, never written to disk in the image) for a first working version.
    RunPod also supports referencing console-managed Secrets from a template's env vars
    instead of literal values — check the current RunPod docs/console for the exact syntax
    when you want the container to never receive the raw value from this script at all.
    """
    image_ref = _image_ref(tag)

    registry_auth = runpod.create_container_registry_auth(
        name=f"ghcr-{os.environ['GITHUB_USERNAME']}-{int(time.time())}",  # unique: names collide
        username=os.environ["GITHUB_USERNAME"],
        password=os.environ["GHCR_TOKEN"],
    )
    logger.info(f"Registry auth: {registry_auth}")

    template = runpod.create_template(
        name=f"{TEMPLATE_NAME}-{int(time.time())}",
        image_name=image_ref,
        is_serverless=True,
        container_disk_in_gb=20,
        registry_auth_id=registry_auth["id"],
        env={
            "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
            "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", ""),
            # entrypoint.sh uses this to clone/pull the private repo at container start instead
            # of baking application code into the image (see Dockerfile) — code/config changes
            # take effect on the next worker spin-up with no rebuild.
            "GITHUB_REPO_TOKEN": os.environ.get("GITHUB_REPO_TOKEN", ""),
            # train_gnn.py's init_wandb() auto-disables (no hang, no crash) when this is absent —
            # only set it if you want serverless runs to actually log to wandb.
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        },
    )
    logger.info(f"Template: {template}")

    endpoint = runpod.create_endpoint(
        name=ENDPOINT_NAME,
        template_id=template["id"],
        gpu_ids=gpu_ids,
        workers_min=0,      # scale-to-zero when idle
        workers_max=4,      # account quota is 5 total workers across all endpoints; the first
                            # (now-stale) endpoint already reserved 1, leaving 4 for this one
        idle_timeout=30,
    )
    logger.info(f"Endpoint: {endpoint}")
    return endpoint["id"]


def invoke(endpoint_id: str, job_input: dict, timeout: int = 3600, poll_interval: int = 10) -> dict:
    """Submit a job and block for the result (like Modal's `.remote()`).

    Uses async dispatch (`Endpoint.run()`) + polling `.status()`/`.output()`, NOT
    `Endpoint.run_sync()` — confirmed by direct diagnosis that run_sync silently returns None for
    jobs longer than a few minutes (RunPod's synchronous /runsync HTTP endpoint appears to have
    its own connection-level ceiling shorter than a full PaySim training run needs, ~10-15 min,
    even though the job itself keeps running server-side and completes fine). Every PaySim
    full-graph job dispatched via run_sync failed this way; Elliptic's small/fast jobs (1-3 min)
    never hit it, which is why this went unnoticed for a while. Polling has no such ceiling."""
    endpoint = runpod.Endpoint(endpoint_id)
    job = endpoint.run(job_input)
    start = time.time()
    status = job.status()
    while status in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() - start > timeout:
            raise TimeoutError(f"Job {job.job_id} did not complete within {timeout}s (status={status})")
        time.sleep(poll_interval)
        status = job.status()
    output = job.output()
    if status != "COMPLETED":
        raise RuntimeError(f"Job {job.job_id} finished with status={status}: {output}")
    return {"id": job.job_id, "status": status, "output": output}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build")

    deploy_p = sub.add_parser("deploy")
    deploy_p.add_argument("--tag", default="latest")

    invoke_p = sub.add_parser("invoke")
    invoke_p.add_argument("--endpoint-id", required=True)
    invoke_p.add_argument("--config", default="configs/paysim.yaml")
    invoke_p.add_argument("--no-preprocess", action="store_true")

    args = parser.parse_args()
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    if args.command == "build":
        build_and_push_image()
    elif args.command == "deploy":
        endpoint_id = deploy(tag=args.tag)
        print(f"\nSave this: --endpoint-id {endpoint_id}")
    elif args.command == "invoke":
        result = invoke(
            args.endpoint_id,
            {"config": args.config, "preprocess": not args.no_preprocess},
        )
        # Deliberately print(), not logger.info() — this is the command's actual output (the job
        # result), not a status/progress message, and may be piped/captured by a caller.
        print(result)


if __name__ == "__main__":
    main()
