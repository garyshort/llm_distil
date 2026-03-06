# Demo 4: LoRA Training on Azure ML

Same task as demo2 (insurance damage extraction) but runs on **Azure Machine Learning** with PyTorch, Hugging Face Transformers, and PEFT LoRA. Uses the same JSONL data format from demo2.

## Prerequisites

- Azure subscription with an ML workspace
- GPU compute cluster (e.g. `Standard_NC6s_v3` or similar)
- Data: `demo2/data/train.jsonl`, `demo2/data/valid.jsonl` (from demo2)

## Local Training (optional)

To run training locally (requires a GPU):

```bash
cd demos
pip install -r requirements.txt
cd demo4
python train.py --data-dir ../demo2/data --output-dir ./outputs
```

For a ~2 minute conference demo (0.5B model, 20 steps):

```bash
python train.py --data-dir ../demo2/data --output-dir ./outputs --demo
```

## Azure ML Job Submission

1. **Configure Azure credentials** – Add to `demos/.env`:

   ```
   AZURE_SUBSCRIPTION_ID=<your-subscription-id>
   AZURE_RESOURCE_GROUP=<your-resource-group>
   AZUREML_WORKSPACE_NAME=<your-workspace-name>
   ```

2. **Create a GPU compute cluster** in Azure ML Studio (e.g. `gs-distil-compute` with NC-series VMs).

3. **Submit the job**:

   ```bash
   cd demos
   pip install -r requirements.txt
   python demo4/submit_job.py
   ```

   For a ~2 minute conference demo:

   ```bash
   python demo4/submit_job.py --demo
   ```

   Or with custom compute/experiment:

   ```bash
   python demo4/submit_job.py --compute gs-distil-compute --experiment demo4-damage
   ```

4. **Monitor** in Azure ML Studio. The job uploads `demo2/data` and runs `train.py`. Adapters are saved to the job outputs.

## Files

- `train.py` – LoRA fine-tuning script (Qwen2.5-7B, PEFT, prompt-masked loss)
- `submit_job.py` – Submits `train.py` as an Azure ML command job
- `environment.yml` – Conda environment for the Azure ML job
- Root `demos/requirements.txt` – Python deps (includes demo4)

## Model

- **Full training**: `Qwen/Qwen2.5-7B-Instruct` (~3–4 h on Standard_NC6s_v3)
- **Demo mode** (`--demo`): `Qwen/Qwen2.5-0.5B-Instruct` (~2 min on Standard_NC6s_v3)
- **LoRA**: r=16, alpha=32, target modules: q_proj, k_proj, v_proj, o_proj
- **Output**: LoRA adapters saved to `outputs/` (can be used with Transformers for inference)
