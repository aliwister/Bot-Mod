#!/usr/bin/env python3
"""
Generate comments for every row in a JSON file using random LLM assignment.

Usage:
  python generate_comment_llm_messages.py comments-id-matched.json --outdir ../dataset/id
      -> ../dataset/id/comments-id-matched-generated.json  (random LLM per row)

  python generate_comment_llm_messages.py comments-ood-matched.json --outdir ../dataset/ood --all-llms
      -> ../dataset/ood/comments-ood-matched-generated-qwen.json
      -> ../dataset/ood/comments-ood-matched-generated-mistral.json
      -> ../dataset/ood/comments-ood-matched-generated-llama.json
"""
import argparse, json, random, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input JSON (e.g. comment-train.json or comment-test.json)")
parser.add_argument("--all-llms", action="store_true",
                    help="Generate one output file per LLM instead of random assignment")
parser.add_argument("--outdir", default=None,
                    help="Directory for output files (default: same directory as input)")
args = parser.parse_args()

import os
INPUT_FILE = args.input
_input_dir  = os.path.dirname(INPUT_FILE) or "."
_input_base = os.path.basename(INPUT_FILE)
stem = _input_base.removeprefix("comments-").removesuffix(".json")
_outdir = args.outdir if args.outdir else _input_dir
OUT_FILE = os.path.join(_outdir, f"comments-{stem}-generated.json")

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------
_BASE_URL = {
    "Qwen/Qwen3-8B":                      "http://localhost:8000/v1",
    "mistralai/Mistral-7B-Instruct-v0.3": "http://localhost:8001/v1",
    "meta-llama/Llama-3.1-8B-Instruct":   "http://localhost:8002/v1",
}
_NOTHINK_MODELS = {"Qwen/Qwen3-8B"}

_clients: dict[str, OpenAI] = {
    url: OpenAI(base_url=url, api_key="", timeout=300.0)
    for url in _BASE_URL.values()
}

def call_llm_messages(messages: list, temperature: float, model: str,
                      max_new_tokens: int = 1024) -> str:
    client = _clients[_BASE_URL[model]]
    response = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


COMMENT_TEMPLATE = '{\n    "content": "<the content of your comment, which should be a direct response to the user\'s post>"\n}'


class UserBot:
    def __init__(self, system_prompt: str, context: str, model: str):
        self.model = model
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]
        suffix = " /nothink" if model in _NOTHINK_MODELS else ""
        prompt = (
            f"Context:\n{context}\n\n"
            f"Comment on the post. Return comment in this format and nothing else: "
            f"{COMMENT_TEMPLATE}{suffix}"
        )
        self.M = self._call(prompt)

    def _call(self, prompt: str, temp: float = 0.3) -> str:
        self._messages.append({"role": "user", "content": prompt})
        response = call_llm_messages(self._messages, temp, self.model)
        self._messages.append({"role": "assistant", "content": response})
        return response.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_ALL_MODELS = list(_BASE_URL.keys())
MODEL_KEY = {
    "Qwen/Qwen3-8B":                      "qwen",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "meta-llama/Llama-3.1-8B-Instruct":   "llama",
}

with open(INPUT_FILE, encoding="utf-8") as f:
    rows = json.load(f)

total = len(rows)


def gen_row(task):
    idx, row, model = task
    community     = row["community"].strip()
    system_prompt = row["system_prompt"].strip()
    context       = json.dumps(row["context"], ensure_ascii=False)
    try:
        user = UserBot(system_prompt, context, model=model)
        text = user.M
    except Exception as e:
        print(f"[{idx+1}/{total}] ERROR: {e}", flush=True)
        text = ""
    print(
        f"[{idx+1}/{total}] model={MODEL_KEY[model]}"
        f"  intent={row['intent']}  label={row['intent_type']}",
        flush=True,
    )
    return {
        "text":          text,
        "community":     community,
        "system_prompt": system_prompt,
        "label_binary":  row["intent_type"].strip().lower(),
        "label_intent":  row["intent"].strip().lower(),
        "mode":          "comment",
        "user_model":    MODEL_KEY[model],
        "context":       row["context"],
    }


t0 = time.time()

if args.all_llms:
    for model in _ALL_MODELS:
        key = MODEL_KEY[model]
        out = os.path.join(_outdir, f"comments-{stem}-generated-{key}.json")
        print(f"\nGenerating {total} comments with {key} → {out}", flush=True)
        tasks = [(i, rows[i], model) for i in range(total)]
        with ThreadPoolExecutor() as ex:
            results = list(ex.map(gen_row, tasks))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} rows to {out}", flush=True)
else:
    random.seed(42)
    assignments = [random.choice(_ALL_MODELS) for _ in rows]
    dist = {}
    for m in assignments:
        dist[MODEL_KEY[m]] = dist.get(MODEL_KEY[m], 0) + 1
    print(f"Generating {total} comments  model distribution: {dist}", flush=True)
    tasks = [(i, rows[i], assignments[i]) for i in range(total)]
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(gen_row, tasks))
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s — wrote {len(results)} rows to {OUT_FILE}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
