# Graph-of-Thoughts Standup Generator

This folder contains a minimal Graph of Thoughts (GoT) system tailored to structured stand-up joke generation. It follows the **Graph of Thoughts** paper (Besta et al., 2023; see `../graph of thoughts.pdf`), which models reasoning as a graph where each *thought* is a vertex and edges capture dependencies. GoT enables branching, scoring, backtracking, and aggregating thoughts; this code now uses those ideas to branch over setups, incongruities, callbacks, and punchlines while letting an evaluator pick the best next category each step.

## High-level flow (GoT perspective)
- **Thought graph**: `GraphReasoningState` stores a directed graph (`networkx.DiGraph`) with the root node describing the start of the routine. Every generated setup or punchline is added as a vertex with typed metadata and an edge from its parent thought.
- **Prompts → Thoughts → Parsing**: `Prompter` builds structured prompts for setups, incongruities, punchlines, callbacks, refinements, aggregation, scoring, and next-category choice; `LLMInterface` implementations generate raw text; `Parser` extracts structured fields (setup/incongruity/callback/punchline + score/pros/cons/reasoning/next/category).
- **Validation + pruning**: `Validation` enforces sanity checks (minimum lengths; numeric scores in 1–10; allowed next-actions/category). Failing thoughts are discarded before they enter the graph.
- **Graph operations (branch, refine, aggregate, score)**: `GraphOfOperations` can branch setups, incongruities, punchlines, callbacks, refine beats, aggregate beats, and score nodes using full-path context. Scores include an explicit `next_action` (refine, branch_setup, branch_punchline, branch_callback, branch_incongruity, backtrack, keep, end).
- **Orchestration**: `Controller.run` scores multiple initial setups, picks the best, then repeatedly asks the evaluator which category fits best next (setup/incongruity/punchline/callback/end), generates three candidates of that category, scores them, and continues with the top choice. Returns the chosen script with per-beat diagnostics and evaluator decisions.

## Module guide (file-by-file)

### controller.py
- **`Controller`**: Orchestrates a full GoT run end-to-end.
  - `__init__(llm: LLMInterface = None)`: Wires together the core components. Defaults to `MockLLM` if no model is passed, instantiates `Prompter`, `Parser`, `Validation`, `GraphOfOperations`, and a fresh `GraphReasoningState`.
  - `run(topic: str, num_branches: int = 3, max_steps: int = 8, min_score: int = 6, k: int = 1) -> List[Dict]`: Scores initial setups, keeps the best path, then loops: ask evaluator for next category (setup/incongruity/punchline/callback/end), generate 3 candidates of that type, score, and advance with the top-scoring beat. Stops on `end`, weak punchlines, or step limit and returns a single script with beats + evaluator decisions.
  - `log_candidates(candidate_ids: List[str], title: str)`: Pretty-prints each candidate node’s ID, content, score, and reasoning from the current `GraphReasoningState`.

### operations.py
- **`GraphOfOperations`**: Encapsulates graph transformations (branching, refinement, aggregation, scoring).
  - `generate_setup(...)` / `generate_followup_setup(...)`: Create initial or follow-up setups/incongruities using topic/path context.
  - `generate_punchline(...)`: Branch punchlines from any node, using nearest setup/incongruity context and sibling punchlines to encourage diversity.
  - `generate_callback(...)`: Craft callbacks anchored to prior beats in the selected path.
  - `refine_node(...)`: Self-loop refinements using cons/feedback; adds refined children.
  - `aggregate_nodes(...)`: Merge multiple nodes into a stronger beat (setup or punchline).
  - `score_candidates(...)`: Full-path contextual scoring with `Score/Pros/Cons/Reasoning/Next`, populating node attributes; `choose_next_category(...)` asks evaluator which beat type comes next.
  - `keep_best(...)`: Utility to pick top-k by score.

### state.py
- **`GraphReasoningState`**: Maintains the evolving thought graph and provides helpers.
  - Constructor initializes a `networkx.DiGraph`, inserts a `root` node with content `"Start of Standup Routine"`, `type="root"`, and `score=0`, plus empty pros/cons/reasoning/next_action.
  - `add_vertex(...)`: Adds typed nodes with score/pros/cons/reasoning/next_action/metadata, and links to parent.
  - `get_vertex(vertex_id) -> Dict`: Direct access to a node’s attribute dictionary.
  - `get_successors(vertex_id) -> List[str]`: Returns outgoing neighbors (children).
  - `get_leaves() -> List[str]`: Returns nodes with out-degree 0 (useful to find current frontier thoughts).
  - `get_path_to_root(...)` / `get_path_contents(...)`: Returns ids/contents along root→vertex for contextual scoring and prompt-building.

### llm_interface.py
- **`LLMInterface` (abstract base)**: Defines the `generate(prompt, max_new_tokens=512, temperature=0.7) -> str` contract used everywhere.
- **`MockLLM`**: Lightweight test double that returns:
  - A random score with canned reasoning when the prompt asks for scoring.
  - Templated setups/incongruities/punchlines/callbacks and category picks for testing traversal.
  This enables deterministic pipeline testing without a real model.
- **`LlamaLLM`**: Concrete implementation using Hugging Face `transformers` for Meta Llama 3.1.
  - Uses the comedy adapter by default (see `../Llama-3.1-Comedy-Adapter-Lables`) and 4-bit quantization for inference.
  - Loads tokenizer/model with chat template, moves to `device_map="auto"`, and applies a fixed system prompt to enforce format.
  - `generate` builds chat-formatted input, applies attention mask, samples with `do_sample=True`, and decodes only new tokens for clean output.

### prompter.py
- **`Prompter`**: Centralized prompt factory to keep all text the LLM sees in one place.
  - Setup/incongruity: `generate_setup_prompt` (topic), `generate_followup_setup_prompt` (context chain + category).
  - Punchlines: `generate_punchline_prompt` (setup/incongruity + prior punchlines).
  - Callbacks: `generate_callback_prompt` (path + anchor).
  - Refinement: `generate_refine_prompt` to polish an existing beat.
  - Aggregation: `generate_aggregate_prompt` to merge beats.
  - Scoring: `score_thought_prompt` requests `Score/Pros/Cons/Reasoning/Next` with a strict rubric and expanded next actions.
  - Next-category choice: `choose_next_category_prompt` asks the evaluator to pick {setup, incongruity, punchline, callback, end}.

### parser.py
- **`Parser`**: Regex-based extractors for structured fields.
  - `parse_setup`, `parse_incongruity`, `parse_callback`, `parse_punchline` pull labeled fields with fallback.
  - `parse_score` and `parse_rich_score`: The latter returns `{"score", "pros", "cons", "reasoning", "next"}`.
  - `parse_category_decision` extracts `{category, reason}` from evaluator picks.

### validation.py
- **`Validation`**: Basic guards to reject unusable generations.
  - `validate_setup(setup)`: Rejects empty setups or those shorter than 10 characters.
  - `validate_punchline(punchline)`: Rejects empty punchlines or those shorter than 2 characters.
  - `validate_score` / `validate_rich_score`: Rich score requires `score` in 1–10 and `next` in `{refine, branch_setup, branch_punchline, branch_callback, branch_incongruity, backtrack, keep, end}`.
  - `validate_category`: Allows `{setup, incongruity, punchline, callback, end}`.

## Putting it together
- Instantiate a `Controller` (optionally pass `LlamaLLM` for real generations), then call `run(topic, num_branches=3, max_steps=8, min_score=6)`.
- The system scores multiple initial setups, keeps the best, then iterates category selection → candidate generation → scoring → choose best until `end` or limits. Returns a single script with beat-by-beat scores and evaluator decisions.
- Each component is isolated (prompts, parsing, validation, operations, state), so you can further tune traversal logic, thresholds, or add new action types without rewriting the rest.
