# Standup Comedy Generation with Auto-Labeled Script Dataset and GoT Prompting 🎙️

This repository contains code and resources for fine-tuning Large
Language Models (specifically **Llama 3.1 8B**) to analyze the structure
of stand-up comedy routines and generate stylistic scripts. It utilizes
the Hugging Face `transformers` library, `peft` for efficient
fine-tuning (LoRA/QLoRA), and `bitsandbytes` for quantization.
> **Note** Details about GoT system is in ./got/README.md

## 📌 Table of Contents

-   [Environment Setup](#-environment-setup)
-   [Hugging Face Authentication](#-hugging-face-authentication)
-   [Instructions](#-instructions)
    -   [1. Data Preparation](#1-data-preparation)
    -   [2. Training (Fine-tuning)](#2-training-fine-tuning)
    -   [3. Inference](#3-inference)
-   [Reproducibility Notes](#-reproducibility-notes)
-   [License](#license)

------------------------------------------------------------------------

## 🛠 Environment Setup

To ensure a clean installation and avoid dependency conflicts, we
recommend using **Conda**.


### Install Dependencies

Install the required Python packages using pip:

``` bash
pip install -r requirements.txt
```
------------------------------------------------------------------------

## 🔑 Hugging Face Authentication

Accessing gated models (like Meta-Llama-3.1-8B) or pushing models to the
Hub requires authentication.

### Step 1: Generate an Access Token

1.  Go to your Hugging Face **Settings → Access Tokens**.\
2.  Click **New token**.\
3.  Select a role:
    -   **Read** if only downloading models\
    -   **Write** if pushing fine-tuned models\
4.  Copy the token (starts with `hf_...`).

### Step 2: Login via Terminal

```bash
hf auth login
```

### Step 3: Input Token

When prompted:

    Enter your token (input will not be visible):

Paste your token and press Enter.

For non-interactive or CI usage, you can provide the token via stdin.
- Windows (cmd):

```cmd
echo %HF_TOKEN% | hf auth login --stdin
```

- PowerShell:

```powershell
echo $env:HF_TOKEN | hf auth login --stdin
```

**Important:**\
You must also visit the **Meta Llama 3.1** model page and accept the
license agreement to access weights. 

------------------------------------------------------------------------

## 🚀 Instructions

### 1. Data Preparation

Ensure your raw transcripts are placed in the `data/` directory.

- **Formats:** Plain text (one transcript per line) or CSV/TSV. The included labeling script expects `data/transcript.txt` by default (one sample per line; optional leading ID separated by tab).
- **Labeling:** Use the provided labeling tool to generate the JSONL training file.

Labeling example (process all samples):

```bash
# From repository root
python label/label.py
```

For quick tests, set the `MAX_SAMPLES` variable inside `label/label.py` to a small integer, or run using an environment variable override (bash):

```bash
# process first 10 samples by temporarily overriding MAX_SAMPLES
python -c "import label.label as L; L.MAX_SAMPLES=10; L.process_transcripts()"
```

The labeling script writes `train/labeled_dataset.jsonl` (one JSON object per line). This is the file used by the training scripts.

------------------------------------------------------------------------

### 2. Training (Fine-tuning)

This repository contains helper scripts under the `train/` directory.

- Labeling / dataset preparation:

```bash
python train/train_label.py --help
```

Run the labeling script to produce `train/labeled_dataset.jsonl` (example):

```bash
python train/train_label.py --input ./train/labeled_dataset.jsonl --output ./checkpoints
```

If you have a fine-tuning entrypoint (for example `train.py`), point it at the labeled dataset produced above. Example QLoRA-style command (adapt paths/flags to your training script):

```bash
python train.py --model_id "meta-llama/Meta-Llama-3.1-8B" --data_path "./train/labeled_dataset.jsonl" --output_dir "./checkpoints" --num_epochs 3 --batch_size 4 --learning_rate 2e-4
```

If your project uses a different training pipeline, adapt the dataset path to `./train/labeled_dataset.jsonl`.

------------------------------------------------------------------------

### 3. Inference (run via included shell scripts)

Inference in this repository is exposed through executable shell scripts at the repository root. Use these to run baseline experiments or the GoT controller:

- `run_baseline0.sh`, `run_baseline1.sh`, `run_baseline2.sh` — baseline pipelines
- `run_got.sh` — runs the GoT Controller using `got/` code and the `LlamaLLM` adapter

Make scripts executable (Linux / macOS / Git Bash / WSL):

```bash
chmod +x run_baseline0.sh run_baseline1.sh run_baseline2.sh run_got.sh
```

Examples:

- Run the GoT controller for a single topic:

```bash
./run_got.sh "dating app"
```

- Run the GoT controller for N random topics:

```bash
./run_got.sh 3
```

- Run a baseline script for one or more topics:

```bash
./run_baseline0.sh "polictics in U.S."
```

The shell scripts accept several environment variables to override defaults (see the top of each script):

- MODEL_PATH (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- ADAPTER_PATH (default: `napalna/Llama-3.1-Comedy-Adapter-Lables`)
- QUANTIZATION (default: `4bit`)
- NUM_BRANCHES, MAX_STEPS, MIN_SCORE

Windows note (cmd.exe): use Git Bash, WSL, or call the scripts with `bash`:

```cmd
bash run_got.sh "dating app"
```

Script output is printed to stdout and can be redirected to files. Adjust model and adapter paths via the environment variables above if you want to use different checkpoints or adapters.

------------------------------------------------------------------------

## ♻️ Reproducibility Notes

To ensure experiments are consistent and repeatable:

### 1. Random Seeds

``` python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### 2. Deterministic Operations

``` python
torch.use_deterministic_algorithms(True)
```

Note: This may slightly slow down training.

### 3. Hardware Context

Refining dataset & training were done on:
-    **NVIDIA A100**
-    **NVIDIA RTX 4090**

Experiments were tested on:

-   **NVIDIA RTX 5090**

Different GPUs or CUDA versions (11.8 vs 12.1) may lead to small
numerical differences.

### 4. Library Versions

Use the exact versions from `requirements.txt` to ensure compatibility
with:

-   `transformers`
-   `peft`
-   `bitsandbytes`

------------------------------------------------------------------------

## License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

## Reference

Got System is based on : 
Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Giannazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., & Hoefler, T.  
**Graph of Thoughts: Solving Elaborate Problems with Large Language Models.**  
AAAI Conference on Artificial Intelligence (AAAI 2024).  
ArXiv: [https://arxiv.org/abs/2308.09687](https://arxiv.org/abs/2308.09687)

Raw comedy transcript is from :
https://huggingface.co/datasets/zachgitt/comedy-transcripts



