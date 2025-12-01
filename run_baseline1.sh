#!/usr/bin/env bash
set -euo pipefail

# Run baseline1 generative pipeline for one or more topics.
# Usage:
#   ./run_baseline1.sh "topic one" "topic two"
#   ./run_baseline1.sh 3   # runs 3 random topics from TOPIC_POOL

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
)

usage() {
  echo "Usage: $0 <topic ... | count>"
  echo "  - Pass one or more topics to run directly."
  echo "  - Or pass a single integer N to run N random topics from TOPIC_POOL."
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
$(for t in "${TOPIC_POOL[@]}"; do printf '    "%s",\n' "$t"; done)
]
count = ${count}
for _ in range(count):
    print(random.choice(pool))
PY
)
else
  topics=("$@")
fi

# Join topics with the ASCII unit separator to safely pass into Python
TOPICS_JOINED="$(printf '%s\x1f' "${topics[@]}")"

# Run a single Python process that loads the model once and iterates topics
TOPICS="$TOPICS_JOINED" \
MODEL_PATH="$MODEL_PATH" \
ADAPTER_PATH="$ADAPTER_PATH" \
QUANTIZATION="$QUANTIZATION" \
python - <<PY
import os, json
from got.llm_interface import LlamaLLM

# Reconstruct topics list
raw = os.environ.get('TOPICS', '')
if raw:
    topics = [t for t in raw.split('\x1f') if t]
else:
    topics = []

if not topics:
    print('No topics provided.')
    raise SystemExit(1)

llm = LlamaLLM(
    model_path=os.environ.get('MODEL_PATH'),
    adapter_path=os.environ.get('ADAPTER_PATH'),
    quantization=os.environ.get('QUANTIZATION'),
)

os.makedirs('outputs', exist_ok=True)
for topic in topics:
    print(f"=== Running baseline1 for topic: {topic} ===")
    result = llm.generate(prompt = f"""You are a professional stand-up comedian.
Style: observational
Topic: {topic}

Generate a stand-up comedy script.
Format your output as:
Script: [Your script here]""", max_new_tokens = 2048)

    out = result if isinstance(result, str) else json.dumps(result)

    fname = f"outputs/baseline1_{topic.replace(' ', '_')}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({"topic": topic, "result": out}, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] wrote final joke to {fname}")
    print()
PY
