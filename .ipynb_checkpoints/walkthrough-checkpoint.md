# Walkthrough: Running the GoT Standup System

## Prerequisites
- Python 3.10+ with `pip`.
- CUDA GPU recommended (4bit/8bit loading uses bitsandbytes); CPU-only is possible but slow for Llama 3.1.
- Access to the Meta Llama 3.1 base model and the comedy LoRA (Hugging Face token with access approval).

## Setup
1) From the repo root (`Standup4AI`), install deps (ideally in a venv):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2) Ensure the `got/` folder is on `PYTHONPATH` (running from the repo root already satisfies this).

## Run via helper script (Llama 3.1 + comedy adapter)
1) Authenticate once: `huggingface-cli login` with a token that can pull `meta-llama/Meta-Llama-3.1-8B-Instruct` and `napalna/Llama-3.1-Comedy-Adapter-Labels` (or use local paths).
2) Make the runner executable (one time): `chmod +x run_got.sh`
3) Run with explicit topics (one or many):
   ```bash
   ./run_got.sh "bad corporate icebreakers" "airports with too many outlets"
   ```
4) Or run N random topics from the built-in pool:
   ```bash
   ./run_got.sh 3
   ```
   Each run prints the GoT decisions + script JSON.

### Script knobs (env vars)
- `MODEL_PATH` (default `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `ADAPTER_PATH` (default `napalna/Llama-3.1-Comedy-Adapter-Labels`; set to your local adapter dir if downloaded)
- `QUANTIZATION` (`4bit` default; `8bit` if you hit memory limits)
- `NUM_BRANCHES`, `MAX_STEPS`, `MIN_SCORE` (passed into `Controller.run`)

## Manual Python run (if you prefer inline)
1) Authenticate as above.
2) Launch the GoT pipeline manually:
   ```bash
   python - <<'PY'
from got.controller import Controller
from got.llm_interface import LlamaLLM

llm = LlamaLLM(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    adapter_path="napalna/Llama-3.1-Comedy-Adapter-Labels",
    quantization="4bit",  # or "8bit"; requires GPU for speed/memory
)

controller = Controller(llm)
result = controller.run("bad corporate icebreakers", num_branches=3, max_steps=8, min_score=6)
import json; print(json.dumps(result, indent=2))
PY
   ```
   - First run downloads weights to your Hugging Face cache; later runs reuse them.
   - If you hit CUDA/memory errors, try `quantization="8bit"` or reduce `max_new_tokens` in `LlamaLLM.generate`.

## Key knobs (`Controller.run`)
- `topic`: high-level premise for the standup routine.
- `num_branches`: how many candidates to sample for each step (setups/punchlines/etc.).
- `max_steps`: maximum category-selection iterations.
- `min_score`: stop early if the chosen punchline scores below this.
- `k`: how many top branches to keep when pruning (currently best=1).

## Tips
- To debug candidates and scores, call `controller.log_candidates(ids, title)` after generation steps.
- Graph state lives in-memory (`GraphReasoningState.graph`); persist or visualize with `networkx` if desired (see `NETWORKX_DI_GRAPH.md` for basics).
