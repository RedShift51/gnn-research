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
        "config": "configs/paysim.yaml",   # path baked into the image
        "preprocess": true                  # rebuild the graph, or reuse an existing processed_path
    }
"""

import runpod

from data.paysim_preprocess import load_config, run_from_config as preprocess_run
from training.train_gnn import run_from_config as train_run


def handler(job):
    job_input = job.get("input", {})
    config_path = job_input.get("config", "configs/paysim.yaml")
    do_preprocess = job_input.get("preprocess", True)

    config = load_config(config_path)

    if do_preprocess:
        preprocess_run(config)

    return train_run(config, config_path)


runpod.serverless.start({"handler": handler})
