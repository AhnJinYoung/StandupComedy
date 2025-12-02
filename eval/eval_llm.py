#!/usr/bin/env python3
"""
Evaluate generated stand-up transcripts with OpenRouter (Grok 4.1 fast) and
produce per-topic statistics.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
import httpx
from openai import OpenAI

MODELS = ["baseline0", "baseline1", "baseline2", "got"]
METRICS = [
    "humor_effectiveness",
    "joke_structure",
    "controllability",
    "human_likeness",
    "overall_score",
]


def discover_topics(processed_dir: Path) -> List[str]:
    """Find topics that have transcripts for all four model variants."""
    present: Dict[str, set[str]] = {name: set() for name in MODELS}
    for path in processed_dir.glob("*.txt"):
        name = path.name
        if "_" not in name:
            continue
        prefix, rest = name.split("_", 1)
        if prefix not in present:
            continue
        topic = rest[:-4]  # strip .txt
        present[prefix].add(topic)

    common = set.intersection(*(present[m] for m in MODELS))
    return sorted(common)


def read_transcripts(processed_dir: Path, topic: str) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for model_name in MODELS:
        path = processed_dir / f"{model_name}_{topic}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing transcript for {model_name} at {path}")
        transcripts[model_name] = path.read_text(encoding="utf-8")
    return transcripts


def patch_httpx_for_proxies() -> None:
    """
    Work around httpx 0.28 removing the `proxies` kwarg that OpenAI 1.12 still passes.
    Replaces the OpenAI SyncHttpxClientWrapper with a proxy-compatible subclass.
    """
    sig = inspect.signature(httpx.Client.__init__)
    if "proxies" in sig.parameters:
        return  # compatible version
    if "proxy" not in sig.parameters:
        return  # unexpected signature; do nothing

    from openai import _base_client

    class ProxySafeClient(httpx.Client):
        def __init__(self, *args, proxies=None, proxy=None, **kwargs):
            if proxy is None and proxies not in (None, getattr(_base_client, "NOT_GIVEN", None)):
                proxy = proxies
            super().__init__(*args, proxy=proxy, **kwargs)

        def __del__(self) -> None:
            try:
                self.close()
            except Exception:
                pass

    _base_client.SyncHttpxClientWrapper = ProxySafeClient


def build_prompt(topic: str, transcripts: Dict[str, str]) -> List[Dict[str, str]]:
    """Create chat messages for the OpenRouter call."""
    system_msg = (
        "You are an expert stand-up comedy evaluator. "
        "Score each transcript independently and fairly."
    )

    format_description = {
        "topic": topic,
        "scores": {
            "baseline0": {
                "humor_effectiveness": 0,
                "joke_structure": 0,
                "controllability": 0,
                "human_likeness": 0,
                "overall_score": 0,
                "brief_explanation": "",
            },
            "baseline1": {},
            "baseline2": {},
            "got": {},
        },
    }

    instructions = f"""
Evaluate four stand-up comedy transcripts for the topic "{topic}".

Scoring (1-10 where 10 is excellent):
1) Humor Effectiveness: How funny is it? Do the jokes land?
2) Joke Structure: Setup → punchline → callbacks. Is there a clear arc?
3) Controllability: Does it stay on topic and follow instructions?
4) Human-likeness: Does it sound authentic and natural?
5) Overall Score: Your overall assessment.

Return ONLY a JSON object exactly matching this shape:
{json.dumps(format_description, ensure_ascii=True, indent=2)}

Use 2-3 sentences per transcript for "brief_explanation".
"""
    transcript_blocks = []
    for model_name in MODELS:
        transcript_blocks.append(f"## Transcript: {model_name}\n{transcripts[model_name]}")

    user_msg = instructions + "\n\n" + "\n\n".join(transcript_blocks)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_openrouter(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Any:
    """Send a chat completion request and return the full response."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                # No max_tokens to avoid truncating long contexts.
                extra_body={"reasoning": {"enabled": True}},
                timeout=60,
            )
            return resp
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay)
    raise RuntimeError(f"OpenRouter request failed after {max_retries} retries: {last_error}")


def sanitize_scores(raw_scores: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure numeric fields are floats in [1,10] and keep brief_explanation."""
    cleaned: Dict[str, Dict[str, Any]] = {}
    for model_name in MODELS:
        model_scores = raw_scores.get(model_name, {})
        normalized: Dict[str, Any] = {}
        for metric in METRICS:
            value = model_scores.get(metric)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            numeric = max(1.0, min(10.0, numeric))
            normalized[metric] = numeric
        explanation = model_scores.get("brief_explanation", "")
        if isinstance(explanation, str):
            normalized["brief_explanation"] = explanation.strip()
        cleaned[model_name] = normalized
    return cleaned


def aggregate_stats(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate mean raw scores across topics per model."""
    totals_raw: Dict[str, Dict[str, List[float]]] = {
        name: {metric: [] for metric in METRICS} for name in MODELS
    }

    for item in results:
        scores = item["scores"]
        for model_name in MODELS:
            for metric in METRICS:
                raw_val = scores.get(model_name, {}).get(metric)
                if raw_val is not None:
                    totals_raw[model_name][metric].append(float(raw_val))

    summary: Dict[str, Any] = {}
    for model_name in MODELS:
        summary[model_name] = {"raw_mean": {}}
        for metric in METRICS:
            raw_vals = totals_raw[model_name][metric]
            if raw_vals:
                summary[model_name]["raw_mean"][metric] = sum(raw_vals) / len(raw_vals)
    return summary


def save_interaction(io_dir: Path, topic: str, model: str, messages: List[Dict[str, str]], response: Any) -> None:
    """Persist request/response pairs for audit."""
    io_dir.mkdir(parents=True, exist_ok=True)
    request_path = io_dir / f"{topic}_request.json"
    response_path = io_dir / f"{topic}_response.json"
    request_path.write_text(
        json.dumps({"topic": topic, "model": model, "messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        response_payload = response.model_dump()
    except AttributeError:
        try:
            response_payload = response.to_dict()
        except AttributeError:
            response_payload = response  # best-effort
    try:
        response_text = json.dumps(response_payload, ensure_ascii=False, indent=2)
    except TypeError:
        response_text = json.dumps({"raw_repr": repr(response_payload)}, ensure_ascii=False, indent=2)
    response_path.write_text(response_text, encoding="utf-8")


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def save_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write flat CSV with one row per topic/model."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["topic", "model"] + METRICS + ["brief_explanation"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            topic = rec["topic"]
            for model_name in MODELS:
                row = {"topic": topic, "model": model_name}
                scores = rec["scores"].get(model_name, {})
                for metric in METRICS:
                    row[metric] = scores.get(metric)
                row["brief_explanation"] = scores.get("brief_explanation", "")
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score stand-up transcripts via OpenRouter Grok 4.1 fast."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("outputs/processed"),
        help="Directory containing *_<topic>.txt files.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("eval/llm_scores.jsonl"),
        help="Where to store per-topic LLM scores.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("eval/llm_stats.json"),
        help="Where to store aggregated statistics.",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=Path("eval/llm_scores.csv"),
        help="Where to store flat CSV of all scores.",
    )
    parser.add_argument(
        "--io-dir",
        type=Path,
        default=Path("eval/llm_io"),
        help="Directory to store raw request/response pairs.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free"),
        help="OpenRouter model id (default: x-ai/grok-4.1-fast:free).",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        help="Optional list of topics to score (default: all common topics).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of topics for a quick test.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between OpenRouter calls.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY environment variable.")

    patch_httpx_for_proxies()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    topics = args.topics or discover_topics(args.processed_dir)
    if args.limit:
        topics = topics[: args.limit]

    records: List[Dict[str, Any]] = []
    for idx, topic in enumerate(topics, start=1):
        transcripts = read_transcripts(args.processed_dir, topic)
        messages = build_prompt(topic, transcripts)
        raw_response = call_openrouter(client, args.model, messages)
        save_interaction(args.io_dir, topic, args.model, messages, raw_response)
        raw_content = raw_response.choices[0].message.content
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse JSON for topic '{topic}': {raw_content}") from exc

        raw_scores = parsed.get("scores", {})
        scores = sanitize_scores(raw_scores)
        record = {
            "topic": topic,
            "scores": scores,
        }
        records.append(record)
        print(f"[{idx}/{len(topics)}] Scored topic '{topic}'")
        time.sleep(args.sleep)

    summary = aggregate_stats(records)
    save_jsonl(args.out_file, records)
    save_csv(args.csv_file, records)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved per-topic scores to {args.out_file}")
    print(f"Saved per-topic scores CSV to {args.csv_file}")
    print(f"Saved summary stats to {args.summary_file}")


if __name__ == "__main__":
    main()
