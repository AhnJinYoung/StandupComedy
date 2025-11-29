import re
from typing import Dict, Any, Optional


class Parser:
    """Parses LLM outputs."""

    def _parse_labeled(self, text: str, label: str) -> str:
        pattern = rf"{label}:\s*(.*)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def parse_setup(self, text: str) -> str:
        """Extracts the setup from the text."""
        return self._parse_labeled(text, "Setup")

    def parse_incongruity(self, text: str) -> str:
        """Extracts an incongruity line."""
        return self._parse_labeled(text, "Incongruity")

    def parse_callback(self, text: str) -> str:
        """Extracts a callback line."""
        return self._parse_labeled(text, "Callback")

    def parse_punchline(self, text: str) -> str:
        """Extracts the punchline from the text."""
        return self._parse_labeled(text, "Punchline")

    def parse_score(self, text: str) -> Dict[str, Any]:
        """Extracts score and reasoning."""
        score_match = re.search(r"Score:\s*(\d+)", text)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else 0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return {"score": score, "reasoning": reasoning}

    def parse_rich_score(self, text: str) -> Dict[str, Any]:
        """Extracts score, pros/cons, reasoning, and next action."""
        score_match = re.search(r"Score:\s*(\d+)", text)
        pros_match = re.search(r"Pros:\s*(.*)", text)
        cons_match = re.search(r"Cons:\s*(.*)", text)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)
        next_match = re.search(r"Next:\s*([a-zA-Z_]+)", text)

        return {
            "score": int(score_match.group(1)) if score_match else 0,
            "pros": pros_match.group(1).strip() if pros_match else "",
            "cons": cons_match.group(1).strip() if cons_match else "",
            "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
            "next": next_match.group(1).strip().lower() if next_match else "",
        }

    def parse_category_decision(self, text: str) -> Dict[str, str]:
        """Parses the evaluator's category pick for the next beat."""
        category_match = re.search(r"Category:\s*([a-zA-Z_]+)", text)
        reason_match = re.search(r"Reason:\s*(.*)", text, re.DOTALL)
        return {
            "category": category_match.group(1).strip().lower() if category_match else "",
            "reason": reason_match.group(1).strip() if reason_match else "",
        }
