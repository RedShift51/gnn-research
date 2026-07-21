"""RunPod Serverless worker entrypoint — the serverless analogue of a Modal `@app.function`.

Where Modal wraps a plain Python function and handles deploy/invoke transparently, RunPod
Serverless is lower-level: this file runs *inside* the container on every job, and a separate
management script (infra/runpod_serverless.py) builds/pushes the image and creates/invokes the
endpoint from your machine. `runpod.serverless.start({"handler": handler})` is what turns this
file into a worker process that pulls jobs off RunPod's queue.

NOTE: this package is named `serverless/`, not `runpod/` — a local `runpod/` package would shadow
the installed `runpod` PyPI SDK we import below (both cwd and the SDK's install dir end up on
sys.path when run as `-m serverless.handler`), breaking `import runpod` on itself.

Job input shape (see infra/runpod_serverless.py for how jobs are submitted):
    {
        "config": "configs/paysim.yaml",   # path baked into the image, OR
        "config_dict": {...},               # a full config dict inline — no image rebuild needed
                                             # to try new hyperparameters, just change the payload
        "preprocess": true                  # rebuild the graph, or reuse an existing processed_path
    }
Exactly one of "config"/"config_dict" should be set. "config" still exists for configs that are
genuinely part of the repo (checked in, reused across runs); "config_dict" is for one-off sweeps
where committing a new YAML file just to trigger a rebuild would be pure overhead.
"""

import logging
import os
import subprocess

import runpod

from data.download import ensure_downloaded, ensure_downloaded_elliptic
from data.elliptic_preprocess import run_from_config as elliptic_preprocess_run
from data.paysim_preprocess import load_config, run_from_config as paysim_preprocess_run
from training.train_gnn import run_from_config as train_run

# Configured once at worker-process import time (not inside handler()) — this process stays
# alive across many jobs (see _git_pull_latest()'s docstring), so basicConfig here covers every
# job the worker ever handles, not just the first.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _git_pull_latest() -> str:
    """Re-sync the repo checkout at the START of every job, not just once at container boot.
    RunPod reuses the same long-lived worker process across many sequential jobs far more
    eagerly than expected — entrypoint.sh's one-time git pull never runs again for that worker,
    so it silently drifts behind (see LAB_JOURNAL.md: three different jobs in one night hit
    FileNotFoundError for configs added hours earlier, on three different "warm" workers). This
    fixes config-file staleness. It does NOT fix stale in-memory *code* — already-imported
    Python modules aren't reloaded — but every failure so far has been a missing config file."""
    try:
        subprocess.run(["git", "-C", "/root/repo", "pull", "--ff-only"],
                       check=True, capture_output=True, text=True, timeout=30)
        return subprocess.run(["git", "-C", "/root/repo", "rev-parse", "HEAD"],
                               check=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        logger.warning(f"git pull at job start failed (continuing with existing checkout): {e}")
        return os.environ.get("GIT_COMMIT_SHA", "unknown")


def handler(job):
    commit_sha = _git_pull_latest()
    job_input = job.get("input", {})
    # Log the raw job input FIRST, before any resolution/defaulting logic runs — this is the
    # earliest possible point to catch a wrong/stale-worker misconfiguration (see LAB_JOURNAL.md's
    # caught 40%-oversample incident, where a stale worker silently ignored config_dict and ran a
    # completely different experiment with no error).
    logger.info(f"Received job_input: {job_input}")
    logger.info(f"Worker git commit (after per-job pull): {commit_sha}")

    config_dict = job_input.get("config_dict")
    do_preprocess = job_input.get("preprocess", True)

    if config_dict is not None:
        config = config_dict
        # config_path is only used for the journal's "config=" label — there's no real file for
        # an inline config, so label it with the run_name instead of the misleading default path.
        config_label = f"inline:{config.get('journal', {}).get('run_name', 'unnamed')}"
    else:
        config_path = job_input.get("config", "configs/paysim.yaml")
        config = load_config(config_path)
        config_label = config_path

    logger.info(f"Resolved config_label: {config_label}")
    logger.info(f"Resolved config: {config}")

    dataset = config.get("data", {}).get("dataset", "paysim")
    if do_preprocess:
        # The image ships no dataset (.dockerignore excludes data/raw/) — download it into the
        # container on first use. entrypoint.sh already wrote ~/.kaggle/kaggle.json from the
        # KAGGLE_USERNAME/KAGGLE_KEY env vars (RunPod Secrets or template env) before this runs.
        if dataset == "elliptic":
            ensure_downloaded_elliptic()
            elliptic_preprocess_run(config)
        elif dataset == "paysim":
            ensure_downloaded()
            paysim_preprocess_run(config)
        else:
            raise ValueError(f"Unknown data.dataset: {dataset!r} (expected 'paysim' or 'elliptic')")

    if config.get("diffusion"):
        # Train a TabDDPM on fraud-only TRAIN features, sample synthetic fraud nodes, and add
        # them to the graph before GNN training. This is the actual diffusion-augmentation
        # research contribution, not just another GNN baseline — see LAB_JOURNAL.md.
        from data.augment_graph import run_from_config as augment_run
        from training.train_diffusion import run_from_config as diffusion_train_run

        logger.info("config.diffusion present — training TabDDPM + augmenting graph before GNN training")
        diffusion_train_run(config)
        augment_run(config)
        config["data"]["processed_path"] = config["data"]["augmented_processed_path"]
        logger.info(f"GNN training will use the augmented graph: {config['data']['processed_path']}")

    result = train_run(config, config_label, git_commit=commit_sha)
    # In the returned output (not just logs) so a stale worker is detectable from the job status
    # API alone, even in the case where it ISN'T missing a file (e.g. only a few commits behind).
    result["worker_git_commit"] = commit_sha
    return result


runpod.serverless.start({"handler": handler})
