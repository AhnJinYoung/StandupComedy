# Standup Comedy Analysis & Generation with Llama 3.1 🎙️

This repository contains code and resources for fine-tuning Large
Language Models (specifically **Llama 3.1 8B**) to analyze the structure
of stand-up comedy routines and generate stylistic scripts. It utilizes
the Hugging Face `transformers` library, `peft` for efficient
fine-tuning (LoRA/QLoRA), and `bitsandbytes` for quantization.

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

### 1. Create and Activate Environment

``` bash
conda create -n comedy python=3.10
conda activate comedy
```

### 2. Install Dependencies

Install the required Python packages using pip:

``` bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet available, install the core libraries
manually:

``` bash
pip install torch transformers datasets accelerate bitsandbytes huggingface_hub peft
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

``` bash
huggingface-cli login
```

### Step 3: Input Token

When prompted:

    Enter your token (input will not be visible):

Paste your token and press Enter.

**Important:**\
You must also visit the **Meta Llama 3.1** model page and accept the
license agreement to access weights.

------------------------------------------------------------------------

## 🚀 Instructions

### 1. Data Preparation

Ensure your dataset is placed in the `data/` directory.

-   **Formats:** CSV or JSONL\
-   **Required column:** `text` containing the transcript or analysis
    target

------------------------------------------------------------------------

### 2. Training (Fine-tuning)

This project uses **QLoRA** for memory-efficient fine-tuning.

``` bash
python train.py     --model_id "meta-llama/Meta-Llama-3.1-8B"     --data_path "./data/dataset.csv"     --output_dir "./checkpoints"     --num_epochs 3     --batch_size 4     --learning_rate 2e-4
```

------------------------------------------------------------------------

### 3. Inference

``` bash
python inference.py     --base_model "meta-llama/Meta-Llama-3.1-8B"     --lora_adapter "./checkpoints/final_adapter"     --prompt "Analyze the set-up and punchline of the following joke:"
```

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

Experiments were tested on:

-   **NVIDIA RTX 3090**
-   **A100**

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
