#!/usr/bin/env python3
"""
Generate posts for every row in a CSV using random LLM assignment.

Usage:
  python generate_post_llm_messages.py raw-gpt-generated-sys-prompts/dataset-id-judge_filtered.csv \\
      --outdir ../dataset/id
      -> ../dataset/id/posts-dataset-id-judge_filtered-generated.json  (random LLM per row)

  python generate_post_llm_messages.py raw-gpt-generated-sys-prompts/dataset-ood-judge_filtered.csv \\
      --outdir ../dataset/ood --all-llms
      -> ../dataset/ood/posts-dataset-ood-judge_filtered-generated-qwen.json
      -> ../dataset/ood/posts-dataset-ood-judge_filtered-generated-mistral.json
      -> ../dataset/ood/posts-dataset-ood-judge_filtered-generated-llama.json

Options:
  --outdir DIR   Directory for output files (default: same directory as input)
"""
import argparse, csv, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input CSV (e.g. train.csv or test.csv)")
parser.add_argument("--all-llms", action="store_true",
                    help="Generate one output file per LLM plus a combined file")
parser.add_argument("--outdir", default=None,
                    help="Directory for output files (default: same directory as input)")
args = parser.parse_args()

INPUT_FILE  = args.input
_input_dir  = os.path.dirname(INPUT_FILE) or "."
_input_base = os.path.basename(INPUT_FILE)
stem        = _input_base.removesuffix(".csv")
_outdir     = args.outdir if args.outdir else _input_dir
OUT_FILE    = os.path.join(_outdir, f"posts-{stem}-generated.json")

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


POST_TEMPLATE = """{
    "submolt": "<name-of-community>",
    "title": "<post-title>",
    "content": "<post body text with markdown and emojis.>"
}"""


class UserBot:
    def __init__(self, sub: str, system_prompt: str, model: str):
        self.model = model
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]
        suffix = " /nothink" if model in _NOTHINK_MODELS else ""
        prompt = f"Post to {sub}. Return post in this format and nothing else: {POST_TEMPLATE}{suffix}"
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

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

total = len(rows)


def gen_row(task):
    idx, row, model = task
    community     = row["community"].strip()
    system_prompt = row["system_prompt"].strip()
    try:
        user = UserBot(community, system_prompt, model=model)
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
        "mode":          "post",
        "user_model":    MODEL_KEY[model],
    }


t0 = time.time()

if args.all_llms:
    for model in _ALL_MODELS:
        key = MODEL_KEY[model]
        out = os.path.join(_outdir, f"posts-{stem}-generated-{key}.json")
        print(f"\nGenerating {total} posts with {key} → {out}", flush=True)
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
    print(f"Generating {total} posts  model distribution: {dist}", flush=True)
    tasks = [(i, rows[i], assignments[i]) for i in range(total)]
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(gen_row, tasks))
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s — wrote {len(results)} rows to {OUT_FILE}", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
