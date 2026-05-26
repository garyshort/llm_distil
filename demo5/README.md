# Demo 5: Vision Distillation — Cats vs Dogs

**What it shows:** Use a strong vision model as a *teacher* to produce
chat-format training data, then optionally fine-tune an open-source
vision-language model (e.g. Qwen2-VL) as a *student*.

The task is binary: classify each image as a cat or a dog. This
intentionally uses a simple, publicly labelled dataset so the
teacher-labelling pipeline is easy to understand and validate.

> **Why teacher labelling?**
> In real-world workflows the same pattern is valuable when labels are
> unavailable, expensive to produce, inconsistent, or need conversion
> into conversational/chat-format supervision suitable for multimodal
> fine-tuning.

---

## Data source

[Oxford-IIIT Pet Dataset](http://robots.ox.ac.uk/~vgg/data/pets/)
— released under CC BY-SA 4.0.

> O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
> *Cats and Dogs*, CVPR 2012.

The dataset contains ~7,400 images of cats and dogs across 37 breeds.

**Oxford labels are used only for image organisation, balanced sampling,
and optional teacher-vs-dataset diagnostics. They are not used as
training labels. The canonical training labels come exclusively from the
teacher vision model.**

---

## Pipeline

```
image + prompt  →  teacher vision model  →  validated JSON  →  training.jsonl
```

1. `create_training_data.py` downloads and extracts the Oxford-IIIT dataset
   (once, ~800 MB).
2. `annotations/list.txt` is parsed to build balanced cat/dog image pools.
3. Each sampled image is copied to `dataset/images/{cat|dog}/` with a
   deterministic filename.
4. The image is sent to an Azure OpenAI vision teacher model together with
   `cat_dog_prompt.txt`.
5. The teacher response is strictly validated as `{"animal":"cat"}` or
   `{"animal":"dog"}`.
6. Valid responses are written as chat-format rows to `dataset/training.jsonl`.
7. A metadata sidecar is written to `dataset/training_metadata.jsonl` for
   auditing, debugging, and resume support.

---

## Quick start

```bash
cd demos/demo5

# Dry-run: download dataset, build pools, preview rows — no API calls
python create_training_data.py --dry-run --max-per-class cat:5,dog:5

# Small real run
python create_training_data.py --max-per-class cat:10,dog:10

# Balanced 500/100 split, reproducible
python create_training_data.py --max-per-class cat:500,dog:100 --seed 42

# Resume after interruption
python create_training_data.py --max-per-class cat:500,dog:100 --resume

# Use OPENAI_* endpoint instead of GPT54_ENDPOINT_*
python create_training_data.py --max-per-class cat:5,dog:5 --use-openai-endpoint
```

---

## Environment variables

Set the following in `demos/.env`:

```
GPT54_ENDPOINT_API_KEY=...
GPT54_ENDPOINT_BASE_URL=https://your-resource.cognitiveservices.azure.com/
GPT54_ENDPOINT_API_VERSION=2024-12-01-preview
GPT54_ENDPOINT_MODEL=your-deployment-name
```

Or use the standard OpenAI endpoint with `--use-openai-endpoint`:

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

No Azure Storage credentials are required. Images are downloaded directly
from the Oxford-IIIT public URL.

---

## Output layout

```
dataset/
├── training.jsonl          # chat-format rows for Qwen fine-tuning
├── training_metadata.jsonl # per-image audit trail; doubles as resume manifest
├── images/
│   ├── cat/cat_000001.jpg ...
│   └── dog/dog_000001.jpg ...
└── source/
    └── oxford-iiit-pet/    # downloaded once; safe to delete and re-download
        ├── images/
        └── annotations/
```

All output under `dataset/` is gitignored. Register the `dataset/` folder
as an Azure ML Data Asset for training jobs, or copy it to a blob container.

---

## training.jsonl row format

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/cat/cat_000001.jpg"},
        {"type": "text", "text": "<contents of cat_dog_prompt.txt>"}
      ]
    },
    {"role": "assistant", "content": "{\"animal\":\"cat\"}"}
  ]
}
```

Image paths are dataset-relative. The training pipeline must convert them
to `file://` URIs (or absolute paths) before invoking Qwen.

---

## Metadata fields

`training_metadata.jsonl` records per-image provenance:

| Field | Description |
|---|---|
| `source_image` | Original Oxford filename |
| `source_dataset` | `oxford-iiit-pet` |
| `source_dataset_class` | Oxford-assigned label (cat/dog) — organisation only |
| `teacher_label` | Label from teacher model (used for training) |
| `label_source_for_training` | Always `teacher` |
| `teacher_dataset_match` | Whether teacher and Oxford labels agree (debug) |
| `local_image_path` | Dataset-relative path to copied image |
| `prompt_file` / `prompt_hash` | Prompt traceability |
| `raw_model_output` | Verbatim teacher response |
| `retry_count` / `status` / `error_message` | Error traceability |

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--max-per-class` | `1000` | Images per class: integer or `cat:N,dog:N` |
| `--dataset-dir` | `dataset` | Output root directory |
| `--prompt` | `cat_dog_prompt.txt` | Prompt file (relative to script) |
| `--seed` | `42` | Sampling seed for reproducibility |
| `--resume` | off | Skip already-completed images |
| `--dry-run` | off | No API calls; verifies dataset and previews rows |
| `--use-openai-endpoint` | off | Use `OPENAI_*` vars instead of `GPT54_ENDPOINT_*` |
| `--max-retries` | `3` | Retries per image on API or validation failure |
