import json
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from baselines.bert.bert_train import (
    MODEL_NAME, BATCH_SIZE, BotDataset, BertModerator, collate, evaluate,
)

CHECKPOINT = "bert_best.pt"

files = sys.argv[1:] or [
    "cache/comments-test-generated-llama.json",
    "cache/comments-test-generated-mistral.json",
    "cache/comments-test-generated-qwen.json",
    "cache/posts-test-generated-llama.json",
    "cache/posts-test-generated-mistral.json",
    "cache/posts-test-generated-qwen.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/comments-ood-comments-generated-llama.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/comments-ood-comments-generated-mistral.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/comments-ood-comments-generated-qwen.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/posts-ood-posts-generated-llama.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/posts-ood-posts-generated-mistral.json",
    "/home/aha112/bot-moderator/dataset-curation/final/ood/generated-model/posts-ood-posts-generated-qwen.json",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = BertModerator().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Loaded {CHECKPOINT}")

for path in files:
    rows = json.load(open(path))
    ds = BotDataset(rows, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    evaluate(model, loader, device, path.split("/")[-1].replace(".json", ""))
