#!/usr/bin/env bash
set -euo pipefail

# Run the GoT Controller with LlamaLLM for one or more topics.
# Usage:
#   ./run_got.sh "topic one" "topic two"
#   ./run_got.sh 3   # runs 3 random topics

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-napalna/Llama-3.1-Comedy-Adapter-NoLables}"
QUANTIZATION="${QUANTIZATION:-4bit}"
NUM_BRANCHES="${NUM_BRANCHES:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
MIN_SCORE="${MIN_SCORE:-6}"

TOPIC_POOL=(
  "bad corporate icebreakers"
  "airports with too many outlets"
  "zoom calls that never end"
  "AI that apologizes too much"
  "dating app small talk"
  "gym influencers filming everything"
  "tiny rental kitchens"
  "open office seating wars"
  "overly honest smart fridges"
  "self-checkout chaos"
  "lost luggage adventures"
  "first dates gone wrong"
  "overenthusiastic baristas"
  "airport security small talk"
  "gym mirror people"
  "parents using emojis"
  "smart fridges with opinions"
  "wifi passwords in cafes"
  "roommates who label food"
  "office zoom bingo"
  "tech support with parents"
  "overbooked yoga class"
  "elevator small talk"
  "driving test fails"
  "overly honest toddlers"
)

usage() {
  echo "Usage: $0 <topic ... | count>"
  echo "  - Pass one or more topics to run directly."
  echo "  - Or pass a single integer N to run N random topics."
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

declare -a topics
if [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]]; then
  count="$1"
  mapfile -t topics < <(python - <<PY
import random

pool = [
    "bad corporate icebreakers",
    "airports with too many outlets",
    "zoom calls that never end",
    "AI that apologizes too much",
    "dating app small talk",
    "gym influencers filming everything",
    "tiny rental kitchens",
    "open office seating wars",
    "overly honest smart fridges",
    "self-checkout chaos",
    "lost luggage adventures",
    "first dates gone wrong",
    "overenthusiastic baristas",
    "airport security small talk",
    "gym mirror people",
    "parents using emojis",
    "smart fridges with opinions",
    "wifi passwords in cafes",
    "roommates who label food",
    "office zoom bingo",
    "tech support with parents",
    "overbooked yoga class",
    "elevator small talk",
    "driving test fails",
    "overly honest toddlers",
]
count = ${count}
choices = random.sample(pool, k=min(count, len(pool)))
if count > len(pool):
    choices.extend(random.choices(pool, k=count-len(pool)))
for t in choices:
    print(t)
PY
)
else
  topics=("$@")
fi

for topic in "${topics[@]}"; do
  echo "=== Running Baseline 1 for topic: $topic ==="
  python - <<PY
import json
import os
import sys
import re
from got.llm_interface import LlamaLLM
from pathlib import Path

topic = """$topic"""

llm = LlamaLLM(
    model_path="${MODEL_PATH}",
    adapter_path="${ADAPTER_PATH}",
    quantization="${QUANTIZATION}",
)

result = llm.generate(prompt = f"""You are a professional stand-up comedian.
Style: observational
Topic: {topic}

Generate a stand-up comedy script.
Format your output as:
Script: [Your script here]""", max_new_tokens = 2048)

print(json.dumps(result, indent=2))

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/processed", exist_ok=True)

def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_") or "topic"

base = f"baseline1_{slugify(topic)}"
json_path = Path("outputs") / f"{base}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

sys.path.append(str(Path("outputs").resolve()))
from post_process import baseline_post_process  # noqa: E402

txt_path = Path("outputs/processed") / f"{base}.txt"
baseline_post_process(str(json_path), str(txt_path))

print(f"[SAVE] wrote final joke JSON to {json_path}")
print(f"[SAVE] wrote processed TXT to {txt_path}")
PY
  echo
done
