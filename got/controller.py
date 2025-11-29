from typing import List, Dict
import os
import re
from datetime import datetime
from .state import GraphReasoningState
from .operations import GraphOfOperations
from .llm_interface import LLMInterface, MockLLM, LlamaLLM
from .prompter import Prompter
from .parser import Parser
from .validation import Validation
import json


class Controller:
    """
    Controller for the Graph of Thoughts system.
    Orchestrates the generation process.
    """

    def __init__(self, llm: LLMInterface = None):
        self.llm = llm if llm else MockLLM()
        self.prompter = Prompter()
        self.parser = Parser()
        self.validation = Validation()
        self.goo = GraphOfOperations(self.llm, self.prompter, self.parser, self.validation)
        self.state = GraphReasoningState()

    def run(self, topic: str, num_branches: int = 3, max_steps: int = 100, min_score: int = 6, k: int = 1):
        """
        Flexible GoT process:
        1) Branch initial setups and keep the best by score.
        2) Ask evaluator for next category (setup, incongruity, punchline, callback, end) using full script.
        3) Generate candidates of that category (default 3), score, and advance with the best.
        """
        print(f"Starting GoT for topic: {topic}")
        # Fresh state per run
        self.state = GraphReasoningState()

        # Step 1: generate initial setups and keep the top one
        setup_ids = self.goo.generate_setup(self.state, self.state.root_id, topic, num_samples=num_branches)
        self.goo.score_candidates(self.state, setup_ids)
        if not setup_ids:
            return []
        best_setup = self.goo.keep_best(self.state, setup_ids, k=1)[0]
        selected_path = [best_setup]
        self._log_selected_node(best_setup)
        current_id = best_setup

        steps = 0
        decisions = []
        while steps < max_steps:
            script_lines = self.state.get_path_contents(current_id)
            if script_lines and script_lines[0] == "Start of Standup Routine":
                script_lines = script_lines[1:]
            decision = self.goo.choose_next_category(script_lines)
            if not decision.get("category"):
                break
            if decision["category"] == "end":
                decisions.append(decision)
                break

            category = decision["category"]
            decisions.append(decision)

            candidates = []
            if category == "setup":
                candidates = self.goo.generate_followup_setup(self.state, current_id, num_samples=num_branches, category="setup")
            elif category == "incongruity":
                candidates = self.goo.generate_followup_setup(self.state, current_id, num_samples=num_branches, category="incongruity")
            elif category == "punchline":
                candidates = self.goo.generate_punchline(self.state, current_id, num_samples=num_branches)
            elif category == "callback":
                candidates = self.goo.generate_callback(self.state, current_id, num_samples=num_branches)

            if not candidates:
                break

            self.goo.score_candidates(self.state, candidates)
            best_next = self.goo.keep_best(self.state, candidates, k=1)
            if not best_next:
                break

            current_id = best_next[0]
            self._log_selected_node(current_id)
            selected_path.append(current_id)

            node_score = self.state.get_vertex(current_id).get("score", 0)
            if node_score < min_score and category == "punchline":
                # If the payoff is weak, stop early.
                break
            steps += 1

        return [self._export_script(topic, selected_path, decisions)]
    def _slugify(self, text: str) -> str:
        """Create a filesystem-friendly slug."""
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_") or "topic"

    def _save_result(self, topic: str, payload: Dict):
        """Persist final joke JSON to outputs/ folder."""
        os.makedirs("outputs", exist_ok=True)
        fname = f"outputs/got_{topic}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] wrote final joke to {fname}")

    def _export_script(self, topic: str, path_ids: List[str], decisions: List[Dict]) -> Dict:
        """Package the chosen path into a consumable structure."""
        beats = []
        for pid in path_ids:
            node = self.state.get_vertex(pid)
            beats.append(
                {
                    "type": node["type"],
                    "text": node["content"],
                    "score": node.get("score", 0),
                    "pros": node.get("pros", ""),
                    "cons": node.get("cons", ""),
                    "reasoning": node.get("reasoning", ""),
                }
            )
        payload = {"topic": topic, "script": beats, "decisions": decisions}
        self._save_result(topic, payload)
        self._print_transcript(beats)
        self._visualize_graph(path_ids)
        return payload

    def log_candidates(self, candidate_ids: List[str], title: str):
        """Logs candidate vertices for research and debugging."""
        print(f"\n--- {title} ---")
        for cid in candidate_ids:
            node = self.state.get_vertex(cid)
            print(f"ID: {cid}")
            print(f"Content: {node['content']}")
            print(f"Score: {node.get('score', 'N/A')}")
            print(f"Reasoning: {node.get('reasoning', 'N/A')}")
            print("-" * 20)
        print("-------------------\n")

    def _log_selected_node(self, node_id: str):
        """Print the currently selected node with its score."""
        node = self.state.get_vertex(node_id)
        payload = {
            "event": "selected",
            "id": node_id,
            "type": node["type"],
            "content": node["content"],
            "score": node.get("score"),
            "pros": node.get("pros", ""),
            "cons": node.get("cons", ""),
            "reasoning": node.get("reasoning", ""),
        }
        print(json.dumps(payload))

    def _visualize_graph(self, path_ids: List[str]):
        """Print a lightweight view of the graph and the selected path."""
        def short(node_id: str) -> str:
            node = self.state.get_vertex(node_id)
            return f"{node_id[:6]}:{node['type']}"

        # Selected path
        path_str = " -> ".join(short(pid) for pid in path_ids)
        print(f"[PATH] {path_str}")

        # Graph edges overview (id:type)
        print("[GRAPH]")
        for parent, child in self.state.graph.edges():
            print(f"  {short(parent)} -> {short(child)}")

    def _print_transcript(self, beats: List[Dict]):
        """Print the final transcript as plain text (no JSON)."""
        lines = [b["text"] for b in beats if b.get("text")]
        transcript = "\n".join(lines)
        print("[TRANSCRIPT]")
        print(transcript)
