class Validation:
    """Validates generated content."""

    def validate_setup(self, setup: str) -> bool:
        """Checks if the setup is valid."""
        if not setup or len(setup) < 10:
            return False
        return True

    def validate_punchline(self, punchline: str) -> bool:
        """Checks if the punchline is valid."""
        if not punchline or len(punchline) < 2:
            return False
        return True

    def validate_score(self, score_data: dict) -> bool:
        """Checks if the score is valid."""
        if not isinstance(score_data.get("score"), int):
            return False
        if score_data["score"] < 1 or score_data["score"] > 10:
            return False
        return True

    def validate_rich_score(self, score_data: dict) -> bool:
        """Validates rich scoring output."""
        if not self.validate_score(score_data):
            return False
        if not isinstance(score_data.get("pros", ""), str):
            return False
        if not isinstance(score_data.get("cons", ""), str):
            return False
        if not isinstance(score_data.get("reasoning", ""), str):
            return False
        if score_data.get("next", "") not in {
            "refine",
            "branch_setup",
            "branch_punchline",
            "branch_callback",
            "branch_incongruity",
            "backtrack",
            "keep",
            "end",
        }:
            return False
        return True

    def validate_category(self, category: str) -> bool:
        """Validates the next category recommendation."""
        return category in {"setup", "incongruity", "punchline", "callback", "end"}
