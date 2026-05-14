# Moltbook Moderation: Uncovering Hidden Intent Through Multi-Turn Dialogue

This repository accompanies the paper
[**Moltbook Moderation: Uncovering Hidden Intent Through Multi-Turn Dialogue**](https://arxiv.org/abs/2605.12856)
(Al-Lawati, Tripto, Ansari, Lucas, Wang, Lee; 2026). It contains the code and data
splits for the *Bot-Moderation* framework, a lightweight, prompt-only pipeline that
detects malicious bot posts and comments on a synthetic social platform (*Moltbook*)
and classifies their underlying intent into one of five categories. The moderator
engages the target agent through multi-turn dialogue guided by Gibbs-based sampling;
it wraps a single 8B-parameter open-weights LLM (Qwen3-8B) and uses inference only —
no fine-tuning is performed.

The trained / tuned version of this system lives on branch
[`autoresearch/may5`](../../tree/autoresearch/may5) at commit
[`27f6221`](../../commit/27f6221).

---

## Release summary

- **Task.** Two-level classification of posts/comments: `benign / malicious`, plus one
  of five intent classes (`subtle_promotion`, `narrative_pushing`, `spam`, `elicitation`,
  `organic_contribution`). Comments are seen together with the parent post title.
- **Metric.** `val_f1 = f1_binary**0.7 * f1_categorical**0.3`, where `f1_categorical` is
  macro-F1 over the four malicious intents on true-positive items only.
- **Model.** Qwen3-8B served via vLLM at `localhost:8000`. No training.
- **Final pipeline.** Zero-shot self-consistency seed → *k* rounds of interactive
  probing with voted refinement → high-sample final vote.
- **Code footprint.** ~125 lines in [train.py](train.py). The evaluation harness in
  [prepare.py](prepare.py) is fixed and not modified.

---

## Quick start

### Requirements

- Python ≥ 3.10, managed with `uv`. See [pyproject.toml](pyproject.toml).
- GPUs for serving the three open-weights models used by the pipeline and evaluator
  (Qwen3-8B on port 8000, Mistral-7B-Instruct-v0.3 on 8001, Llama-3.1-8B-Instruct on
  8002).
- The `cache/dataset-train.json` and `cache/test-generated-new.json` files, produced by
  the data pipeline that ships with the upstream dataset release.
- A `.env` file in the repo root with any secrets the vLLM server needs (e.g.
  `HF_TOKEN` for gated weights). The start-up scripts below `source` it.

### Install

```bash
uv sync
```

### Start the vLLM servers

The moderator talks to Qwen3-8B; the evaluator's *user-simulator* additionally uses
Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct. Each model must be reachable at
the exact host / port declared in the `_BASE_URL` table in [prepare.py](prepare.py):

| Model | URL |
| --- | --- |
| `Qwen/Qwen3-8B` | `http://localhost:8000/v1` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `http://localhost:8001/v1` |
| `meta-llama/Llama-3.1-8B-Instruct` | `http://localhost:8002/v1` |

Any OpenAI-compatible server will work — we used vLLM. A minimal launcher for one
model looks like this:

```bash
export HF_TOKEN=...                  # required for gated Llama / Mistral weights
export CUDA_VISIBLE_DEVICES=<gpu-ids> # pick free GPU(s) for this model
.venv/bin/vllm serve <model-id> \
    --tensor-parallel-size <N> \
    --host 0.0.0.0 \
    --port <port>
```

Substitute the `<model-id>` / `<port>` pair from the table above, set
`--tensor-parallel-size` to the number of GPUs you gave the process, and — if you
are serving Qwen3-8B and want its reasoning channel parsed — add
`--reasoning-parser qwen3`. Repeat the recipe three times (in three shells, `tmux`
panes, or `nohup ... &` background jobs) so that all three servers run concurrently.

Each server prints `Uvicorn running on http://0.0.0.0:<port>` once it is ready;
wait for all three before running the evaluators.

### Check out the trained configuration

All of the numbers below were produced from commit `27f6221` on branch
`autoresearch/may5`. To reproduce them exactly:

```bash
git fetch origin
git checkout autoresearch/may5
git reset --hard 27f6221
```

(If you already cloned fresh, `git checkout 27f6221` by itself is enough to end up in
a detached-HEAD on the right commit.)

### Train-set evaluation (the loop we optimised)

```bash
python3 train.py
```

This runs the moderator over the train split and prints the metrics block:

```
val_f1:           0.7100
f1_binary:        0.7466
f1_categorical:   0.6316
val_f1_zs:        0.5764
f1_zs:            0.6477
f1_cat_zs:        0.4390
f1_posts:         0.7809
f1_comments:      0.6302
total_seconds:    391.6
```

The `_zs` rows are the moderator's zero-shot pass (no probing); the unlabelled rows
are the full pipeline with two probing iterations and a final high-sample vote.

### Test-set evaluation

```bash
python3 eval.py
```

Uses the identical `ModeratorBot` class against the held-out
`cache/test-generated-new.json`.

### End-to-end one-shot

For a clean machine, the full sequence from checkout to metrics is:

```bash
git clone <this-repo> autoresearch && cd autoresearch
git checkout 27f6221              # pin to the released commit on autoresearch/may5
uv sync                            # install Python deps into .venv
cp /path/to/env .env               # provide HF_TOKEN etc.

# launch each vLLM server (see "Start the vLLM servers" above for the exact
# `vllm serve ...` command, GPU assignment, and port for each model).
# Qwen3-8B on :8000, Mistral-7B-Instruct-v0.3 on :8001, Llama-3.1-8B-Instruct on :8002.

# wait until each server prints "Uvicorn running on http://0.0.0.0:<port>"

python3 train.py   # train-split metrics
python3 eval.py    # held-out test-split metrics
```

---

## Repository layout

| Path | Role |
| --- | --- |
| [train.py](train.py) | `ModeratorBot` — the only file modified during research |
| [prepare.py](prepare.py) | Fixed constants, data loading, LLM client, metric, evaluator |
| [eval.py](eval.py) | Held-out test evaluation (uses the same `ModeratorBot`) |
| [program.md](program.md) | Autonomous-research protocol used during development |
| `cache/` | Pre-generated train/test splits (not tracked) |
| `logs/`, `results.tsv` | Per-experiment logs and results table (not tracked) |

Inside `cache/`:

- `dataset-train.json` — the canonical train split used by [prepare.py](prepare.py).
- `dataset-train-AR.json` — the same split with every entry duplicated (each row
  appears twice). Swapping this file in during the autonomous-research loop halves
  the per-run variance from the LLM at the cost of roughly 2× wall-clock time, which
  makes small effects easier to distinguish from noise when deciding whether to keep
  an experiment.

---

## What changed in this release

Relative to the un-modified baseline `train.py`, the final pipeline adds or changes:

1. **Self-consistency seed** — the zero-shot intent is drawn by majority vote over
   5 samples at temperature 0.7 instead of a single greedy call.
2. **Voted refinement per probing round** — the intent is re-voted (5 samples, t=0.7)
   after each new probe/response pair.
3. **High-sample final vote** — after the last probing round, a final 11-sample vote
   determines the label.
4. **Interactive probing without a critique step** — probes are generated directly from
   the content and previous exchange; the earlier separate "critique" LLM call was
   removed after we observed it was redundant once the rest of the pipeline matured.
5. **Prompt hygiene** — a vigilant/skeptical framing on the *intent* prompt only, a
   compact `Q:` / `A:` probe transcript format, recency-ordered context (content and
   most-recent probe last), and no leakage of the currently-suspected intent into the
   probe-generation prompt.
6. **Deterministic binary mapping** — `organic_contribution → benign`, everything else
   `→ malicious`. Attempts to predict the binary label separately from the intent
   consistently destabilised the system and were removed.

All of the above are visible in [train.py](train.py); each was adopted only after it
improved val_f1 across multiple runs (run-to-run variance is ≈ ±0.015, so every accept
decision required 2–3 reruns).

---

## Reproducibility notes

- Results depend on the vLLM server being reachable and serving the exact model IDs
  declared in [prepare.py](prepare.py). Different serving stacks can change sampling
  behaviour subtly.
- The evaluator uses a `ThreadPoolExecutor(max_workers=64)`; expect 6–8 minutes for a
  full train-split run on a single A10/L4-class GPU behind vLLM.
- `results.tsv` and `logs/` are intentionally untracked — they are regenerated on every
  run.

---

## Citation

If you use this code or the Moltbook moderator pipeline in academic work, please cite:

```bibtex
@misc{allawati2026moltbookmoderationuncoveringhidden,
      title={Moltbook Moderation: Uncovering Hidden Intent Through Multi-Turn Dialogue}, 
      author={Ali Al-Lawati and Nafis Tripto and Abolfazl Ansari and Jason Lucas and Suhang Wang and Dongwon Lee},
      year={2026},
      eprint={2605.12856},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.12856}, 
}
```

## License

Released under the terms stated at the top of the repository. LLM weights are governed
by their upstream licences (Qwen3-8B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct).
