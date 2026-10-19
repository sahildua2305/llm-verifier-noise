import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import modal
from modal.config import config

app = modal.App("llm-verifier-noise")

hf_cache = modal.Volume.from_name(
    "hf-cache", create_if_missing=True
)

runs = modal.Volume.from_name(
    "verifier-runs", create_if_missing=True
)

vllm_logs = modal.Volume.from_name(
    "vllm-logs", create_if_missing=True
)

eval_results = modal.Volume.from_name(
    "verifier-eval-results",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "build-essential",
        "clang",
    )
    .add_local_dir(
        ".",
        "/root/llm-verifier-noise",
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            ".venv-1/**",
            "__pycache__/**",
            "debug_logs/**",
            "eval_results/**",
        ],
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        'pip install "numpy==1.26.4" "pyarrow==25.0.0"',
        'pip install "trl==0.27.1"',
        "cd /root/llm-verifier-noise && pip install -e .",
    )
)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name(
            "huggingface",
            required_keys=["HF_TOKEN"],
        ),
        modal.Secret.from_name(
            "wandb",
            required_keys=["WANDB_API_KEY"],
        )
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/runs": runs,
        "/root/llm-verifier-noise/vllm_logs": vllm_logs,
        "/root/llm-verifier-noise/results": eval_results,
    },
)
def train(
    config: str,
    max_steps: int = 3,
    save_steps: int = 20,
    mixup_strategy: str = "uniform",
    run_name: str | None = None,
):
    os.chdir("/root/llm-verifier-noise")

    # First just verify the machine.
    subprocess.run(["nvidia-smi"], check=True)

    # Cheap runtime editable install:
    # dependencies are already present, so this does NOT reinstall them.
    subprocess.run(
        ["python", "-m", "pip", "install", "-e", ".", "--no-deps"],
        check=True,
    )

    os.environ["WANDB_PROJECT"] = "llm-verifier-noise"

    cmd = [
        "python",
        "src/train_model.py",
        "--config",
        config,
        "--override",
        f"training_args.max_steps={max_steps}",
        f"training_args.save_steps={save_steps}",
        "training_args.push_to_hub=false",
        "training_args.report_to=wandb",
        f"mixup.strategy={mixup_strategy}",
        "move_to=/runs",
        "hf_username=sahildua2305",
    ]
    if run_name:
        cmd.append(f"run_name={run_name}")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    subprocess.run(
        ["find", "/runs", "-maxdepth", "3", "-type", "d"],
        check=True,
    )

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    secrets=[
        modal.Secret.from_name(
            "huggingface",
            required_keys=["HF_TOKEN"],
        ),
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/runs": runs,
        "/root/llm-verifier-noise/vllm_logs": vllm_logs,
        "/root/llm-verifier-noise/results": eval_results,
    },
)
def evaluate(model_name: str):
    os.chdir("/root/llm-verifier-noise")

    cmd = [
        "python",
        "src/eval_model.py",
        "--benchmarks",
        "rgym_decimal_chain_sum_3_6_3_6_3_6",
        "--model_name",
        model_name,
        "--max_model_length",
        "16384",
        "--vllm_port",
        "8000",
        "--reasoning",
    ]

    print("Running:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        eval_results.commit()
    except subprocess.CalledProcessError:
        from pathlib import Path

        log_dir = Path("/root/llm-verifier-noise/vllm_logs")
        logs = sorted(
            log_dir.glob("*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if logs:
            latest = logs[0]
            print(f"\n===== vLLM LOG: {latest} =====\n")
            print(latest.read_text(errors="replace"))
            print("\n===== END vLLM LOG =====\n")

        raise
    finally:
        vllm_logs.commit()


@app.local_entrypoint()
def evaluate_and_download(model_name: str, destination: str = "eval_results"):
    """Evaluate on Modal, then download the complete results volume locally."""
    evaluate.remote(model_name)

    local_destination = Path(destination).resolve()
    local_destination.mkdir(parents=True, exist_ok=True)
    environment = config.get("environment")
    with tempfile.TemporaryDirectory(prefix="eval-results-") as staging_dir:
        cmd = [sys.executable, "-m", "modal", "volume", "get"]
        if environment:
            cmd.extend(["--env", environment])
        cmd.extend(["verifier-eval-results", "/", staging_dir])
        subprocess.run(cmd, check=True)
        shutil.copytree(staging_dir, local_destination, dirs_exist_ok=True)
    print(f"Evaluation results downloaded to: {local_destination}")
