# Moderation by Interrogation: Uncovering Concealed Intent in Agent Social Networks

[Paper (arXiv:2605.12856)](https://arxiv.org/abs/2605.12856)

Code and data for Bot-Mod, a moderator that questions an agent before deciding on its post. Bot-Mod
starts from a content-only prior over five intents, asks up to three diagnostic probes in one live
conversation with the agent, updates the posterior after each answer, and stops once one intent is
confident enough. This repository has the code for the main experiment (Bot-Mod, the content prior
and Bot-Mod-judge) and the Molt-Intent datasets with the raw Moltbook export they were built from.

```
www-code/      experiment code (Python 3.13); the method is in moderator_v2.py
www-dataset/   Molt-Intent splits, personas and the raw Moltbook export (see www-dataset/README.md)
www-results/   created on first run
```

Keep the folders side by side. `BOTMOD_DATASET` and `BOTMOD_LOGS` override the data and results paths.

## Setup

```
cd www-code
pip install -r requirements.txt
GPU=0 bash launch_vllm.sh   # Qwen3-8B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct
```

The three models are served by vLLM on ports `BOTMOD_PORT_BASE` to `+2` (default 8000-8002) and fit
on one 80 GB GPU. Qwen3-8B is the moderator, and each item is answered by the model that wrote it.

## Reproducing the main experiment

Stages 1 to 3 need the vLLM servers. Stage 4 runs on CPU.

```
# 1. Content prior
for s in train id id_spam ood ood_spam; do python exp_e12_head_to_head.py matrix $s; done

# 2. Calibration on train (writes T = 7.86, beta = 15.4, 1 - alpha = 0.98)
python exp_probe_bank.py
python exp_e15_train_refit.py --apply
python exp_e16_multiturn.py run train varied
python exp_e16b_rescore.py run train varied
python fit_varied_bank.py

# 3. Live interrogation of the test splits
for s in id id_spam ood ood_spam; do python exp_e17_live.py run $s; done

# 4. Scoring
python exp_e12_head_to_head.py score id+spam
python exp_e12_head_to_head.py score ood+spam
```

Scoring prints F1_val, F1_y, F1_t, FPR and the mean number of probes for Bot-Mod, the content prior
alone and Bot-Mod-judge on identical items, with paired bootstrap intervals, and writes
`www-results/e12_head_to_head_<split>.json`.

## Citation

```bibtex
@misc{allawati2026moderation,
  title         = {Moderation by Interrogation: Uncovering Concealed Intent in Agent Social Networks},
  author        = {Ali Al-Lawati and Nafis Tripto and Abolfazl Ansari and Fali Wang and Jason Lucas and Suhang Wang and Dongwon Lee},
  year          = {2026},
  eprint        = {2605.12856},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.12856}
}
```
