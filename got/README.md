# Graph-of-Thoughts Standup Generator

This folder contains a compact Graph-of-Thoughts (GoT) system designed to generate structured stand-up routines (setups, incongruities, punchlines, callbacks) and score/branch them automatically.

Key points:

- The main orchestrator is `Controller` in `controller.py`. Call `Controller(llm).run(topic, num_branches, max_steps, min_score)` to produce a single script together with per-beat diagnostics.
- `llm_interface.LlamaLLM` is the production LLM backend (or use `MockLLM` for testing). `LlamaLLM` loads the base model and — optionally — a LoRA adapter. It is intended to be instantiated once and reused for multiple generations.
- Prompt construction is centralized in `prompter.py`; parsing and extraction live in `parser.py`; graph transformations (branch/refine/aggregate/score) are implemented in `operations.py`; graph state is managed by `state.py` and validated with `validation.py`.

How the provided scripts run this code:

- `run_got.sh` (repository root) is the convenience entrypoint for the GoT controller. It accepts either explicit topics or a single integer N to run the first N topics from the `TOPIC_POOL` defined near the top of the script. The script was updated to load the model once and iterate topics in a single Python process to avoid repeated model loads.
- `TOPIC_POOL` is defined in `run_got.sh`; editing that list changes which default topics are used and their order.
- Each run writes raw JSON results into `outputs/` (filenames like `got_<topic>.json`).

Post-processing:

- The generator and evaluator produce some diagnostic fields and structured artifacts used during search. `outputs/post_process.py` (and the README in the repository root) include a simple post-processing utility that aggregates the generated beats into a cleaned text script and strips small formatting artifacts. To post-process all generated outputs, run in the `outputs/` directory:

```bash
cd outputs
python post_process.py
```