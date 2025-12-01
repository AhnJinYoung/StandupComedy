#!/usr/bin/env bash
set -euo pipefail

# Run the GoT Controller with LlamaLLM for one or more topics.
# Usage:
#   ./run_got.sh "topic one" "topic two"
#   ./run_got.sh 3   # runs 3 topics from TOPIC_POOL in order (cycles if N > pool size)

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-napalna/Llama-3.1-Comedy-Adapter-Lables}"
QUANTIZATION="${QUANTIZATION:-4bit}"
NUM_BRANCHES="${NUM_BRANCHES:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
MIN_SCORE="${MIN_SCORE:-6}"

# TOPIC_POOL: edit this list to change default topics (order matters)
TOPIC_POOL=(
  "dating app"
  "self deprecation"
  "polictics in U.S."
)

usage() {
  echo "Usage: $0 <topic ... | count>"
  echo "  - Pass one or more topics to run directly."
  echo "  - Or pass a single integer N to run N topics from TOPIC_POOL in order."
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

declare -a topics
if [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]]; then
  count="$1"
  mapfile -t topics < <(python - <<PY
pool = [
$(for t in "${TOPIC_POOL[@]}"; do printf '    "%s",\n' "$t"; done)
]
count = ${count}
# Emit topics in order, cycling if necessary
for i in range(count):
    print(pool[i % len(pool)])
PY
)
else
  topics=("$@")
fi

# Join topics with the ASCII unit separator to safely pass into Python
TOPICS_JOINED="$(printf '%s\x1f' "${topics[@]}")"

# Run a single Python process that loads the model once and iterates topics
MODEL_PATH_ENV="${MODEL_PATH}"
ADAPTER_PATH_ENV="${ADAPTER_PATH}"
QUANTIZATION_ENV="${QUANTIZATION}"
NUM_BRANCHES_ENV="${NUM_BRANCHES}"
MAX_STEPS_ENV="${MAX_STEPS}"
MIN_SCORE_ENV="${MIN_SCORE}"

TOPICS="$TOPICS_JOINED" \
MODEL_PATH="$MODEL_PATH_ENV" \
ADAPTER_PATH="$ADAPTER_PATH_ENV" \
QUANTIZATION="$QUANTIZATION_ENV" \
NUM_BRANCHES="$NUM_BRANCHES_ENV" \
MAX_STEPS="$MAX_STEPS_ENV" \
MIN_SCORE="$MIN_SCORE_ENV" \
python - <<PY
import os, json
from got.controller import Controller
from got.llm_interface import LlamaLLM

# Reconstruct topics list from the unit separator
raw = os.environ.get('TOPICS', '')
if raw:
    topics = [t for t in raw.split('\x1f') if t]
else:
    topics = []

if not topics:
    print('No topics provided.')
    raise SystemExit(1)

# Load the model once
llm = LlamaLLM(
    model_path=os.environ.get('MODEL_PATH'),
    adapter_path=os.environ.get('ADAPTER_PATH'),
    quantization=os.environ.get('QUANTIZATION'),
)
controller = Controller(llm)

# Parse numeric options
num_branches = int(os.environ.get('NUM_BRANCHES', '2'))
max_steps = int(os.environ.get('MAX_STEPS', '10'))
min_score = int(os.environ.get('MIN_SCORE', '6'))

for topic in topics:
    print(f"=== Running GoT for topic: {topic} ===")
    result = controller.run(
        topic,
        num_branches=num_branches,
        max_steps=max_steps,
        min_score=min_score,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
PY
