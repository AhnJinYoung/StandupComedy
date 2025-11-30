from typing import List, Dict, Any, Callable
import json
import re
from .state import GraphReasoningState
from .llm_interface import LLMInterface
from .prompter import Prompter
from .parser import Parser
from .validation import Validation


class GraphOfOperations:
    """
    Graph of Operations (GoO).
    Defines the operations that can be performed on the graph state.
    """

    def __init__(self, llm: LLMInterface, prompter: Prompter, parser: Parser, validation: Validation):
        self.llm = llm
        self.prompter = prompter
        self.parser = parser
        self.validation = validation
        self._max_log_chars = 1600  # cap log size to stay readable

    def _log_new_node(self, state: GraphReasoningState, vid: str, parent_id: str):
        """Print a JSON line for a freshly added node."""
        node = state.get_vertex(vid)
        score = node.get("score")
        score = None if score == 0 else score
        payload = {
            "event": "gen",
            "id": vid,
            "parent": parent_id,
            "type": node["type"],
            "content": node["content"],
            "score": score,
            "pros": node.get("pros", ""),
            "cons": node.get("cons", ""),
            "reasoning": node.get("reasoning", ""),
            "next_action": node.get("next_action", ""),
        }
        print(json.dumps(payload))

    @staticmethod
    def _log_selection(sorted_candidates):
        """Print candidate scores and selection order as JSON."""
        payload = {
            "event": "select",
            "ordered": [{"id": cid, "score": score} for cid, score in sorted_candidates],
        }
        print(json.dumps(payload))

    @staticmethod
    def _truncate_sentences(text: str, max_sentences: int = 10) -> str:
        """Keep only the first N sentences to keep vertices concise."""
        # Split on sentence end punctuation; fallback to whole text.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(parts) <= max_sentences:
            return text.strip()
        return " ".join(parts[:max_sentences]).strip()

    def _trim(self, text: str) -> str:
        """Trim long prompt/response logs."""
        if len(text) <= self._max_log_chars:
            return text
        return text[: self._max_log_chars] + "... [truncated]"

    def _log_llm_call(self, kind: str, prompt: str, response: str):
        """Log every LLM call with intent and raw response."""
        payload = {
            "event": "llm_call",
            "kind": kind,
            "prompt": self._trim(prompt),
            "response": self._trim(response),
        }
        print(json.dumps(payload))

    def generate_setup(self, state: GraphReasoningState, parent_id: str, topic: str, num_samples: int = 3, style: str = "observational") -> List[str]:
        """Generates candidate setups from root using the high-level topic."""
        candidates = []
        prompt = self.prompter.generate_setup_prompt(topic)

        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            self._log_llm_call("generate_setup", prompt, response)
            setup = self._truncate_sentences(self.parser.parse_setup(response))
            if self.validation.validate_setup(setup):
                vertex_id = state.add_vertex(content=setup, type="setup", parent_id=parent_id)
                self._log_new_node(state, vertex_id, parent_id)
                candidates.append(vertex_id)

        return candidates

    def generate_followup_setup(self, state: GraphReasoningState, parent_id: str, num_samples: int = 2, style: str = "observational", category: str = "setup") -> List[str]:
        """Generate setups/incongruities after any node, conditioned on path context."""
        candidates = []
        transcript_byfar = state.get_path_contents(parent_id)
        prompt = self.prompter.generate_followup_setup_prompt(transcript_byfar, style=style, category=category)
        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            self._log_llm_call(f"generate_{category}", prompt, response)
            if category == "incongruity":
                line = self._truncate_sentences(self.parser.parse_incongruity(response))
            else:
                line = self._truncate_sentences(self.parser.parse_setup(response))
            if self.validation.validate_setup(line):
                vid = state.add_vertex(content=line, type=category, parent_id=parent_id)
                self._log_new_node(state, vid, parent_id)
                candidates.append(vid)
        return candidates

    def generate_punchline(self, state: GraphReasoningState, parent_id: str, num_samples: int = 3) -> List[str]:
        """Generates candidate punchlines branching from any node with a setup ancestor."""
        parent_node = state.get_vertex(parent_id)
        path_ids = state.get_path_to_root(parent_id)
        setup_text = ""
        for pid in reversed(path_ids):
            if state.graph.nodes[pid]["type"] in {"setup", "incongruity"}:
                setup_text = state.graph.nodes[pid]["content"]
                break
        sibling_punches = [state.graph.nodes[c]["content"] for c in state.get_successors(parent_id) if state.graph.nodes[c]["type"] == "punchline"]

        candidates = []
        prompt = self.prompter.generate_punchline_prompt(setup_text or parent_node["content"], prior_punchlines=sibling_punches)
        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            self._log_llm_call("generate_punchline", prompt, response)
            punchline = self._truncate_sentences(self.parser.parse_punchline(response))
            if self.validation.validate_punchline(punchline):
                vid = state.add_vertex(content=punchline, type="punchline", parent_id=parent_id)
                self._log_new_node(state, vid, parent_id)
                candidates.append(vid)
        return candidates

    def generate_callback(self, state: GraphReasoningState, parent_id: str, num_samples: int = 3) -> List[str]:
        """Generate callbacks that lean on earlier beats in the selected path."""
        candidates = []
        transcript_byfar = state.get_path_contents(parent_id)
        anchors = [c for c in transcript_byfar if c != "Start of Standup Routine"]
        if not anchors:
            return candidates
        # Prioritize recent anchors while still considering earlier beats.
        anchor_cycle = anchors[-num_samples:] if len(anchors) >= num_samples else anchors
        for i in range(num_samples):
            anchor = anchor_cycle[i % len(anchor_cycle)]
            prompt = self.prompter.generate_callback_prompt(transcript_byfar, anchor)
            response = self.llm.generate(prompt)
            self._log_llm_call("generate_callback", prompt, response)
            callback_line = self._truncate_sentences(self.parser.parse_callback(response))
            if self.validation.validate_punchline(callback_line):
                vid = state.add_vertex(content=callback_line, type="callback", parent_id=parent_id)
                self._log_new_node(state, vid, parent_id)
                candidates.append(vid)
        return candidates

    def refine_node(self, state: GraphReasoningState, node_id: str, feedback: str = "", num_samples: int = 1) -> List[str]:
        """Refine a node by generating improved variants (self-loop refinement)."""
        node = state.get_vertex(node_id)
        content = node["content"]
        prompt = self.prompter.generate_refine_prompt(content, node["type"], feedback=feedback)
        refined_ids = []
        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            self._log_llm_call("refine", prompt, response)
            if node["type"] in {"setup", "incongruity"}:
                refined = self._truncate_sentences(self.parser.parse_setup(response))
                if self.validation.validate_setup(refined):
                    vid = state.add_vertex(content=refined, type=node["type"], parent_id=node_id)
                    self._log_new_node(state, vid, node_id)
                    refined_ids.append(vid)
            else:
                refined = self._truncate_sentences(self.parser.parse_punchline(response))
                if self.validation.validate_punchline(refined):
                    vid = state.add_vertex(content=refined, type=node["type"], parent_id=node_id)
                    self._log_new_node(state, vid, node_id)
                    refined_ids.append(vid)
        return refined_ids

    def aggregate_nodes(self, state: GraphReasoningState, source_ids: List[str], target_parent: str, target_type: str = "setup") -> str:
        """Aggregate multiple nodes into a single stronger beat."""
        contents = [state.graph.nodes[sid]["content"] for sid in source_ids]
        prompt = self.prompter.generate_aggregate_prompt(contents, target_type=target_type)
        response = self.llm.generate(prompt)
        self._log_llm_call("aggregate", prompt, response)
        if target_type == "setup":
            aggregated = self._truncate_sentences(self.parser.parse_setup(response))
            if not self.validation.validate_setup(aggregated):
                return ""
        else:
            aggregated = self._truncate_sentences(self.parser.parse_punchline(response))
            if not self.validation.validate_punchline(aggregated):
                return ""
        vid = state.add_vertex(content=aggregated, type=target_type, parent_id=target_parent)
        self._log_new_node(state, vid, target_parent)
        return vid

    def score_candidates(self, state: GraphReasoningState, candidate_ids: List[str]):
        """Scores nodes using full-path context and rich schema."""
        for cid in candidate_ids:
            transcript_byfar = state.get_path_contents(cid)
            node = self.get_vertex_safe(state, cid)
            if not node:
                continue
            prompt = self.prompter.score_thought_prompt(transcript_byfar, node["type"])
            response = self.llm.generate(prompt, temperature = 0.2)
            self._log_llm_call("score", prompt, response)
            score_data = self.parser.parse_rich_score(response)
            valid = self.validation.validate_rich_score(score_data)
            if not valid:
                # Relaxed fallback to keep pipeline moving.
                score_data["score"] = score_data.get("score") or 6
                score_data["next"] = score_data.get("next") or "keep"
            state.graph.nodes[cid]["score"] = score_data["score"]
            state.graph.nodes[cid]["pros"] = score_data["pros"]
            state.graph.nodes[cid]["cons"] = score_data["cons"]
            state.graph.nodes[cid]["reasoning"] = score_data["reasoning"]
            state.graph.nodes[cid]["next_action"] = score_data["next"]
            node = state.graph.nodes[cid]
            payload = {
                "event": "score",
                "id": cid,
                "type": node["type"],
                "score": node["score"],
                "next_action": node["next_action"],
                "pros": node["pros"],
                "cons": node["cons"],
                "reasoning": node["reasoning"],
                "valid": valid,
            }
            print(json.dumps(payload))

    def choose_next_category(self, script_lines: List[str]) -> Dict[str, str]:
        """Ask the evaluator for the best next category given the whole script."""
        prompt = self.prompter.choose_next_category_prompt(script_lines)
        response = self.llm.generate(prompt, temperature = 0.2)
        
        self._log_llm_call("choose_next_category", prompt, response)
        decision = self.parser.parse_category_decision(response)
        if not self.validation.validate_category(decision.get("category", "")):
            payload = {
                "event": "decide",
                "category": "",
                "reason": decision.get("reason", ""),
                "note": "invalid category, defaulting to punchline",
            }
            print(json.dumps(payload))
            return {"category": "punchline", "reason": "fallback: invalid decision"}
        payload = {
            "event": "decide",
            "category": decision.get("category", ""),
            "reason": decision.get("reason", ""),
        }
        print(json.dumps(payload))
        return decision

    def keep_best(self, state: GraphReasoningState, candidate_ids: List[str], k: int = 1) -> List[str]:
        """Selects the top k candidates based on score."""
        scored_candidates = []
        for cid in candidate_ids:
            node = state.get_vertex(cid)
            scored_candidates.append((cid, node.get("score", 0)))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        self._log_selection(scored_candidates)
        return [x[0] for x in scored_candidates[:k]]

    @staticmethod
    def get_vertex_safe(state: GraphReasoningState, cid: str) -> Dict[str, Any]:
        """Return vertex dict if present, otherwise {}."""
        try:
            return state.get_vertex(cid)
        except KeyError:
            return {}
