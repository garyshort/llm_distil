# LLM Distillation Demos

A set of demos illustrating the journey from a cloud LLM (GPT) to a locally fine-tuned student model. The task throughout is **insurance damage extraction**: given a narrative description of property damage, extract structured JSON with damage indicators and severity.

---

## Overview

| Demo | What it shows |
|------|---------------|
| **Demo 1** | Call a cloud LLM (OpenAI/Azure GPT) with a prompt + narrative; get structured JSON output |
| **Demo 2** | Generate synthetic training data, train a LoRA adapter on Apple Silicon (MLX), and evaluate |
| **Demo 3** | Run the same prompt as Demo 1 against the locally fine-tuned MLX model (no cloud) |
| **Demo 4** | Same task as Demo 2, but train on **Azure Machine Learning** with PyTorch, Hugging Face, and PEFT |

---

## Prerequisites

- **Python 3.10+**
- **Demo 1**: OpenAI API key (or Azure OpenAI)
- **Demo 2 & 3**: Apple Silicon Mac (MLX runs on M1/M2/M3)
- **Demo 4**: Azure subscription with an ML workspace and GPU compute

---

## Setup

1. **Clone and enter the demos directory**

   ```bash
   cd demos
   ```

2. **Create `.env` from the example**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set at least `OPENAI_API_KEY` for Demo 1. For Demo 4, add Azure credentials.

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   For Demo 2 & 3 (MLX), also install:

   ```bash
   pip install "mlx-lm[train]"
   ```

---

## Demo 1: Cloud LLM (GPT) Extraction

**What it shows:** A baseline extraction using a cloud LLM. You send a prompt template plus a narrative to OpenAI (or Azure OpenAI) and receive structured JSON.

**Run:**

```bash
python demo1/demo1.py
```

**How it works:**

- Loads `shared/demo1_prompt.txt` (the extraction prompt and schema)
- Injects `shared/demo1_narrative.txt` into the `{narrative_text}` placeholder
- Sends the combined prompt to the configured model (`OPENAI_MODEL`, default `gpt-4o-mini`)
- Prints the model’s JSON response

**Environment:** `OPENAI_API_KEY` required. For Azure: set `OPENAI_BASE_URL`, `OPENAI_API_VERSION`, and optionally `OPENAI_MODEL`.

---

## Demo 2: Training Data + MLX LoRA Fine-Tuning

**What it shows:** The full pipeline for creating a locally fine-tuned model on Apple Silicon:

1. **Generate** synthetic training data (narratives + correct JSON)
2. **Split** into train/valid/test
3. **Train** LoRA adapters on Qwen2.5-7B (4-bit quantized) via MLX
4. **Evaluate** on the test set

**Pipeline:**

```bash
cd demo2

# 1. Generate 1000 synthetic prompt/completion pairs
python generate_training_data.py

# 2. Validate the data (optional)
python validate_training_data.py

# 3. Split into train (800) / valid (100) / test (100)
python split_training_data.py

# 4. Train LoRA adapters + evaluate
python train_and_eval_student.py
```

**Quick demo (~2 min):** Use `train_and_eval_student_quick.py` for a shorter run. It uses a separate adapter directory (`adapters_qwen25_7b_damage_demo`) so it does not overwrite the full adapters.

**Output:** LoRA adapters saved to `adapters_qwen25_7b_damage/` (or `adapters_qwen25_7b_damage_demo/` for the quick demo).

---

## Demo 3: Local MLX Inference

**What it shows:** The same extraction task as Demo 1, but using the **locally fine-tuned** MLX model instead of a cloud API. No network calls after the model is loaded.

**Run:**

```bash
python demo3/run.py
```

**Prerequisites:** Run Demo 2 first to produce `demo2/adapters_qwen25_7b_damage/`. Demo 3 loads Qwen2.5-7B with these adapters and runs inference on `shared/demo1_narrative.txt`.

**Output:** Pretty-printed JSON matching the Demo 1 schema.

---

## Demo 4: Azure ML LoRA Training

**What it shows:** The same task as Demo 2 (insurance damage extraction) but training on **Azure Machine Learning** with PyTorch, Hugging Face Transformers, and PEFT LoRA. Uses the same JSONL data format from Demo 2.

**Local training (optional, requires GPU):**

```bash
cd demo4
python train.py --data-dir ../demo2/data --output-dir ./outputs
```

**Quick demo (~2 min):**

```bash
python train.py --data-dir ../demo2/data --output-dir ./outputs --demo
```

**Azure ML job submission:**

1. Add to `demos/.env`:
   ```
   AZURE_SUBSCRIPTION_ID=<your-subscription-id>
   AZURE_RESOURCE_GROUP=<your-resource-group>
   AZUREML_WORKSPACE_NAME=<your-workspace-name>
   ```

2. Create a GPU compute cluster in Azure ML Studio (e.g. `Standard_NC6s_v3`).

3. Submit the job:
   ```bash
   python demo4/submit_job.py
   ```
   For a ~2 min demo: `python demo4/submit_job.py --demo`

4. Monitor in Azure ML Studio. Adapters are saved to the job outputs.

**Models:**

- Full: `Qwen/Qwen2.5-7B-Instruct` (~3–4 h on NC6s_v3)
- Demo: `Qwen/Qwen2.5-0.5B-Instruct` (~2 min)

---

## Shared Resources

| File | Purpose |
|------|---------|
| `shared/demo1_prompt.txt` | Extraction prompt template with schema; `{narrative_text}` is replaced at runtime |
| `shared/demo1_narrative.txt` | Sample narrative used by Demo 1 and Demo 3 |
| `.env` | API keys and config (copy from `.env.example`) |

---

## Project Structure

```
demos/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── shared/
│   ├── demo1_prompt.txt
│   └── demo1_narrative.txt
├── demo1/
│   └── demo1.py              # GPT extraction
├── demo2/
│   ├── generate_training_data.py
│   ├── validate_training_data.py
│   ├── split_training_data.py
│   ├── train_and_eval_student.py
│   ├── train_and_eval_student_quick.py
│   ├── training_data.jsonl    # generated
│   ├── data/                  # train.jsonl, valid.jsonl, test.jsonl
│   └── adapters_qwen25_7b_damage/   # LoRA adapters (generated)
├── demo3/
│   └── run.py                 # Local MLX inference
└── demo4/
    ├── train.py               # PyTorch/PEFT LoRA training
    ├── submit_job.py          # Azure ML job submission
    ├── environment.yml       # Conda env for Azure ML
    └── README.md              # Demo 4 details
```

---

## Schema (Damage Extraction)

All demos use the same JSON schema:

```json
{
  "damage": {
    "broken_plaster": boolean,
    "mould": boolean,
    "floor_water_damage": boolean,
    "electrical_damage": boolean,
    "ceiling_damage": boolean,
    "structural_crack": boolean,
    "carpet_damage": boolean,
    "cabinet_damage": boolean,
    "appliance_damage": boolean,
    "odor_present": boolean
  },
  "overall_severity": "low" | "moderate" | "high"
}
```
