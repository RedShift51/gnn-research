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
import os
import subprocess
import time

import runpod

IMAGE_NAME = "fraud-diffusion-serverless"
TEMPLATE_NAME = "fraud-diffusion-serverless"
ENDPOINT_NAME = "fraud-diffusion-serverless"


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
    print(f"Pushed {image_ref}")
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
    print("Registry auth:", registry_auth)

    template = runpod.create_template(
        name=f"{TEMPLATE_NAME}-{int(time.time())}",
        image_name=image_ref,
        is_serverless=True,
        container_disk_in_gb=20,
        registry_auth_id=registry_auth["id"],
        env={
            "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
            "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", ""),
        },
    )
    print("Template:", template)

    endpoint = runpod.create_endpoint(
        name=ENDPOINT_NAME,
        template_id=template["id"],
        gpu_ids=gpu_ids,
        workers_min=0,      # scale-to-zero when idle
        workers_max=4,      # account quota is 5 total workers across all endpoints; the first
                            # (now-stale) endpoint already reserved 1, leaving 4 for this one
        idle_timeout=30,
    )
    print("Endpoint:", endpoint)
    return endpoint["id"]


def invoke(endpoint_id: str, job_input: dict, timeout: int = 3600) -> dict:
    """Submit a job and block for the result (like Modal's `.remote()`).
    For a job you don't want to block on, use `runpod.Endpoint(endpoint_id).run(job_input)`
    instead and poll `.status()` — useful since training can run long and sync HTTP calls
    have their own timeout on RunPod's side."""
    endpoint = runpod.Endpoint(endpoint_id)
    return endpoint.run_sync(job_input, timeout=timeout)


def main() -> None:
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
        print(result)


if __name__ == "__main__":
    main()
