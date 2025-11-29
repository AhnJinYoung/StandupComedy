# Example GoT Flow (topic: “seeing shit in streets”)

This illustrates how the provided GoT pipeline might behave on the topic “seeing shit in streets”, with plausible mock LLM generations. It follows the Controller sequence: generate setups → score → keep best → generate punchlines → score → keep best.

## 0) Initial state
- Root node: `root` — content: “Start of Standup Routine”, type: `root`, score: `0`.

## 1) Generate setups (branch from root)
Prompts are built by `Prompter.generate_setup_prompt(topic)`.

Sample LLM outputs → parsed setups → inserted nodes:
- `s1`: “Every morning commute I dodge more mysterious sidewalk piles than cars.” (type `setup`, parent `root`)
- `s2`: “My city treats potholes like surprise piñatas, but what’s inside is never candy.” (type `setup`, parent `root`)
- `s3`: “You know you live in a rough area when Google Maps adds ‘watch your step’ as a feature.” (type `setup`, parent `root`)

## 2) Score setups
Each setup is scored via `GraphOfOperations.score_candidates`:
- `s1` score: 7, reasoning: “Clear premise, relatable visual.”
- `s2` score: 6, reasoning: “Funny twist but less direct.”
- `s3` score: 8, reasoning: “Tech reference + urban relatability.”

## 3) Keep best setup(s)
With `k=1`, `keep_best` picks highest score:
- Best setup: `s3`.

## 4) Generate punchlines for best setup
Prompts from `Prompter.generate_punchline_prompt(setup_text)` using `s3`.

Sample punchline candidates (children of `s3`):
- `p1`: “It even reroutes me around a guy arguing with a raccoon over whose trash it is.” (type `punchline`, parent `s3`)
- `p2`: “The walking directions start with ‘Step 1: Accept that something will crunch.’” (type `punchline`, parent `s3`)
- `p3`: “Street View blurs faces, but honestly it should blur the pavement.” (type `punchline`, parent `s3`)

## 5) Score punchlines
Using `score_joke_prompt(setup, punchline)` (setup context included):
- `p1` score: 8, reasoning: “Adds absurd dialogue, strong visual.”
- `p2` score: 7, reasoning: “Darkly playful, but less vivid.”
- `p3` score: 6, reasoning: “Clever but softer laugh.”

## 6) Keep best punchline(s)
- Best punchline: `p1`.

## 7) Final joke output
Returned by `Controller.run`:
```json
[
  {
    "setup": "You know you live in a rough area when Google Maps adds ‘watch your step’ as a feature.",
    "punchline": "It even reroutes me around a guy arguing with a raccoon over whose trash it is.",
    "score": 8
  }
]
```

## How this maps to GoT
- **Branching**: Multiple setups, then multiple punchlines per best setup.
- **Scoring + pruning**: Scores added as node attributes; `keep_best` prunes branches.
- **Graph state**: Nodes (`root`, `s1-3`, `p1-3`) and edges (`root→s*`, `s3→p*`) live in `GraphReasoningState.graph`. Attributes hold `content`, `type`, `score`, `reasoning`.
- **Extensibility**: You could add refining or aggregation steps (per the GoT paper) inside `GraphOfOperations` to loop on weak punchlines or merge ideas from multiple setups before selection.
