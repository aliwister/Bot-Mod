"""Where everything lives, relative to the botmod/ root, so the tree moves as a unit.

    botmod/www-code      this code
    botmod/www-dataset   train / id / ood / id-spam / ood-spam / personas / moltbook-export
    botmod/www-results   every JSON, checkpoint and log the experiments write

BOTMOD_DATASET / BOTMOD_LOGS override the two data roots (used on the GPU server).
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
DATASET = os.environ.get("BOTMOD_DATASET", os.path.join(ROOT, "www-dataset"))
RESULTS = os.environ.get("BOTMOD_LOGS", os.path.join(ROOT, "www-results"))
os.makedirs(RESULTS, exist_ok=True)
TRAIN = os.path.join(DATASET, "train", "dataset-train.json")
