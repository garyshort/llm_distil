#!/usr/bin/env python3
"""
Submit the demo4 training job to Azure Machine Learning.

Requires: pip install azure-ai-ml azure-identity
Environment: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET (or Azure CLI login)
Or use .env with: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME

Usage:
  python submit_job.py
  python submit_job.py --compute gs-distil-compute --experiment demo4-damage
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit demo4 training to Azure ML")
    parser.add_argument(
        "--subscription-id",
        type=str,
        default=None,
        help="Azure subscription ID (or set AZURE_SUBSCRIPTION_ID)",
    )
    parser.add_argument(
        "--resource-group",
        type=str,
        default=None,
        help="Azure resource group (or set AZURE_RESOURCE_GROUP)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Azure ML workspace name (or set AZUREML_WORKSPACE_NAME)",
    )
    parser.add_argument(
        "--compute",
        type=str,
        default="gs-distil-compute",
        help="Compute cluster name (default: gs-distil-compute)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="demo4-damage",
        help="Experiment name",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Local data dir (default: ../demo2/data)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="~2 min conference demo (0.5B model, 20 steps)",
    )
    args = parser.parse_args()

    # Load .env from demos root
    script_dir = Path(__file__).resolve().parent
    demos_root = script_dir.parent
    load_dotenv(demos_root / ".env")

    subscription_id = args.subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = args.resource_group or os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = args.workspace or os.getenv("AZUREML_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        print(
            "Error: Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME\n"
            "  in .env or pass --subscription-id, --resource-group, --workspace",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from azure.ai.ml import Input, MLClient, command
        from azure.ai.ml.entities import Environment
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Install azure-ai-ml and azure-identity: pip install azure-ai-ml azure-identity",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir = (args.data_dir or (demos_root / "demo2" / "data")).resolve()
    if not data_dir.exists():
        print(f"Error: Data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Data dir must exist locally (we're submitting from local machine).
    # Input(type="uri_folder", path=...) tells Azure to UPLOAD this folder and
    # mount it on the compute; ${{inputs.data}} becomes the mount path on Azure.
    train_data_path = data_dir / "train.jsonl"
    valid_data_path = data_dir / "valid.jsonl"
    if not train_data_path.exists() or not valid_data_path.exists():
        print(
            f"Error: train.jsonl and valid.jsonl required in {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create environment with transformers, peft, etc.
    env = Environment(
        name="demo4-damage-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest",
        conda_file=str(script_dir / "environment.yml"),
    )

    # Use ./outputs so Azure ML auto-uploads to Outputs + logs tab (special folder).
    # ${{outputs.output}} goes to blob but may not appear in Studio UI.
    train_cmd = 'python train.py --data-dir "${{inputs.data}}" --output-dir ./outputs'
    if args.demo:
        train_cmd += " --demo"

    desc = "LoRA fine-tune Qwen2.5 for insurance damage extraction"
    if args.demo:
        desc += " (~2 min demo)"

    salt = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = command(
        name="demo4-damage-train" + ("-demo" if args.demo else "") + f"-{salt}",
        display_name="Demo4 LoRA damage extraction" + (" (demo)" if args.demo else ""),
        description=desc,
        experiment_name=args.experiment,
        compute=args.compute,
        code=script_dir,
        command=train_cmd,
        inputs={
            "data": Input(type="uri_folder", path=str(data_dir), mode="ro_mount"),
        },
        environment=env,
    )

    print("Submitting job to Azure ML...")
    try:
        submitted = ml_client.jobs.create_or_update(job)
    except Exception as e:
        print(f"Error submitting job: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Job submitted: {submitted.name}")
    print(f"  Status: {submitted.status}")
    if hasattr(submitted, "studio_url") and submitted.studio_url:
        print(f"  Studio: {submitted.studio_url}")


if __name__ == "__main__":
    main()
