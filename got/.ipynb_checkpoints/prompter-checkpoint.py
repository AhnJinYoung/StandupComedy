from typing import List, Dict, Any


class Prompter:
    """Generates prompts for the LLM."""

    def __init__(self):
        pass

    def generate_setup_prompt(self, topic: str, style: str = "observational") -> str:
        """Generates a prompt for creating a comedy setup."""
        return f"""You are a professional stand-up comedian.
Style: {style}
Topic: {topic}

Generate a setup for a joke in at most 3 sentences. Do not include the punchline yet.
Format your output as:
Setup: [Your setup here]
"""

    def generate_followup_setup_prompt(self, context_chain: List[str], style: str = "observational", category: str = "setup") -> str:
        """Generates a follow-up beat with prior path context."""
        context = "\n".join([f"- {c}" for c in context_chain])
        label = "Incongruity" if category == "incongruity" else "Setup"
        directive = "introduce a surprising left turn that heightens the tension" if category == "incongruity" else "naturally extends the above"
        return f"""You are a professional stand-up comedian building on prior beats.
Style: {style}
Prior beats:
{context}

Write the next {category} line that {directive}, in at most 3 sentences. Keep it tight and stage-ready.
Format:
{label}: [text]
"""

    def generate_punchline_prompt(self, setup: str, prior_punchlines: List[str] = None) -> str:
        """Generates a prompt for creating a punchline given a setup and optional prior punchlines."""
        prior_text = ""
        if prior_punchlines:
            prior_text = "\nPrevious punchlines:\n" + "\n".join([f"- {p}" for p in prior_punchlines])
        return f"""You are a professional stand-up comedian.
Setup: {setup}
{prior_text}

Generate a funny punchline for this setup that adds something new, in at most 3 sentences.
Format your output as:
Punchline: [Your punchline here]
"""

    def generate_callback_prompt(self, path_chain: List[str], anchor: str) -> str:
        """Prompts the model to craft a callback that references an earlier beat."""
        context = "\n".join([f"- {c}" for c in path_chain])
        return f"""You are a stand-up comedian landing a callback.
Script so far:
{context}
Anchor to reference: {anchor}

Write a short callback line that cleverly brings back the anchor without re-explaining it, in at most 3 sentences.
Format:
Callback: [text]
"""

    def generate_refine_prompt(self, content: str, node_type: str, feedback: str = "") -> str:
        """Asks the LLM to refine an existing node (self-loop refinement)."""
        fb = f"Feedback: {feedback}\n" if feedback else ""
        return f"""You are polishing a stand-up {node_type}.
Current {node_type}: {content}
{fb}

Improve it without changing its core premise.
Format:
{node_type.capitalize()}: [revised text]
"""

    def generate_aggregate_prompt(self, contents: List[str], target_type: str = "setup") -> str:
        """Combines multiple thoughts into one stronger beat."""
        bullets = "\n".join([f"- {c}" for c in contents])
        return f"""You are merging several comedy beats into one stronger {target_type}.
Inputs:
{bullets}

Blend the best parts into a single, concise {target_type}.
Format:
{target_type.capitalize()}: [text]
"""

    def score_joke_prompt(self, setup: str, punchline: str) -> str:
        """Generates a prompt for scoring a joke."""
        return f"""Rate the following joke on a scale of 1 to 10 based on humor, structure, and originality.
Setup: {setup}
Punchline: {punchline}

Provide a score and a brief reasoning.
Format your output as:
Score: [1-10]
Reasoning: [Your reasoning]
"""

    def score_thought_prompt(self, path_text: List[str], node_type: str) -> str:
        """Sharper scoring prompt with explicit rubric and deterministic shape."""
        path_fmt = "\n".join([f"{idx+1}. {t}" for idx, t in enumerate(path_text)])
        return f"""You are a stand-up coach. Evaluate the CURRENT BEAT only, given the full script context.
Script so far:
{path_fmt}

Beat type: {node_type}
Score 1-10 (integer only) with these criteria: joke craft, originality, timing/pacing, surprise, relatability, and how well it fits with the chain. Be strict on low-energy or muddled beats.
Return EXACTLY:
Reasoning: [one sentence why you scored it]
Score: [1-10 integer]
Pros: [2 short strengths, comma separated]
Cons: [2 short weaknesses, comma separated]
Next: [refine | branch_setup | branch_punchline | branch_callback | branch_incongruity | backtrack | keep | end]
"""

    def choose_next_category_prompt(self, script_lines: List[str]) -> str:
        """Ask the evaluator to decide the best next category."""
        script = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(script_lines)])
        return f"""You are planning the next beat in a stand-up set. Choose the single best next move.

Script so far:
{script}

Categories:
- setup (lay new premise. If setups have not built up enough pressure, continue constructing setups to build more tension.)
- incongruity (heighten/left turn. If the setups have built up enough pressure, produce a punchline or callback or incongruity)
- punchline (payoff. If the setups have built up enough pressure, produce a punchline or callback or incongruity.)
- callback (bring back an earlier beat. If the setups have built up enough pressure, produce a punchline or callback or incongruity.)
- end (stop here)

Respond EXACTLY:
Reason: [short justification]
Category: [setup|incongruity|punchline|callback|end]
"""
