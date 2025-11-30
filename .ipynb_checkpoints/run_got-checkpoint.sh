#!/usr/bin/env bash
set -euo pipefail

# Run the GoT Controller with LlamaLLM for one or more topics.
# Usage:
#   ./run_got.sh "topic one" "topic two"
#   ./run_got.sh 3   # runs 3 random topics

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-napalna/Llama-3.1-Comedy-Adapter-Lables}"
QUANTIZATION="${QUANTIZATION:-4bit}"
NUM_BRANCHES="${NUM_BRANCHES:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
MIN_SCORE="${MIN_SCORE:-6}"


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
    "dating app",
    "self deprecation",
    "polictics in U.S."
]
count = ${count}
for _ in range(count):
    print(random.choice(pool))
PY
)
else
  topics=("$@")
fi

for topic in "${topics[@]}"; do
  echo "=== Running GoT for topic: $topic ==="
  python - <<PY
import json
from got.controller import Controller
from got.llm_interface import LlamaLLM

topic = """$topic"""

llm = LlamaLLM(
    model_path="${MODEL_PATH}",
    adapter_path="${ADAPTER_PATH}",
    quantization="${QUANTIZATION}",
)

controller = Controller(llm)
result = controller.run(
    topic,
    num_branches=${NUM_BRANCHES},
    max_steps=${MAX_STEPS},
    min_score=${MIN_SCORE},
)
print(json.dumps(result, indent=2))
PY
  echo
done
