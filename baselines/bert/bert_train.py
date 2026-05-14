import copy
import json
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from collections import Counter

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WARMUP_RATIO = 0.1
VAL_FRAC = 0.2
SEED = 42

BINARY_LABELS = ["benign", "malicious"]
INTENT_LABELS = ["organic_contribution", "elicitation", "narrative_pushing", "subtle_promotion", "spam"]


def extract_text(row):
    raw = row["text"].strip()
    community = row.get("community", "")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parts = [str(parsed[k]) for k in ("title", "content") if parsed.get(k)]
            if parts:
                raw = "\n".join(parts)
    except Exception:
        pass
    # prepend parent post title for comments, matching the LLM eval setup
    if row.get("mode") == "comment" and isinstance(row.get("context"), dict):
        parent_title = row["context"].get("post", {}).get("title", "")
        if parent_title:
            raw = f"[POST] {parent_title}\n{raw}"
    return f"[{community}]\n{raw}"


class BotDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            extract_text(row),
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "binary_label": BINARY_LABELS.index(row["label_binary"].strip().lower()),
            "intent_label": INTENT_LABELS.index(row["label_intent"].strip().lower()),
            "mode": row.get("mode", "post"),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "binary_label": torch.tensor([b["binary_label"] for b in batch]),
        "intent_label": torch.tensor([b["intent_label"] for b in batch]),
        "mode": [b["mode"] for b in batch],
    }


class BertModerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.binary_head = nn.Linear(h, len(BINARY_LABELS))
        self.intent_head = nn.Linear(h, len(INTENT_LABELS))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])
        return self.binary_head(cls), self.intent_head(cls)


def class_weights(rows, labels, key):
    counts = Counter(r[key].strip().lower() for r in rows)
    total = sum(counts.values())
    return torch.tensor([total / (len(labels) * counts[l]) for l in labels], dtype=torch.float)


_ALPHA = 0.7
_MAL_INTENT_LABELS = list(range(1, len(INTENT_LABELS)))  # exclude organic_contribution (0)


def _binary_f1(bin_preds, bin_trues):
    tp = sum(1 for p, t in zip(bin_preds, bin_trues) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(bin_preds, bin_trues) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(bin_preds, bin_trues) if p == 0 and t == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _categorical_f1(bin_trues, int_preds, int_trues):
    mal_idx = [i for i, t in enumerate(bin_trues) if t == 1]
    if not mal_idx:
        return 0.0
    return f1_score(
        [int_trues[i] for i in mal_idx],
        [int_preds[i] for i in mal_idx],
        average="macro", labels=_MAL_INTENT_LABELS, zero_division=0,
    )


def evaluate(model, loader, device, split_name):
    model.eval()
    bin_preds, bin_trues, int_preds, int_trues, modes = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            bl, il = model(ids, mask)
            bin_preds += bl.argmax(1).cpu().tolist()
            bin_trues += batch["binary_label"].tolist()
            int_preds += il.argmax(1).cpu().tolist()
            int_trues += batch["intent_label"].tolist()
            modes += batch["mode"]

    f1_bin = _binary_f1(bin_preds, bin_trues)
    f1_cat = _categorical_f1(bin_trues, int_preds, int_trues)
    val_f1 = f1_bin ** _ALPHA * f1_cat ** (1 - _ALPHA)

    def mode_val_f1(idx):
        if not idx:
            return 0.0
        bp = [bin_preds[i] for i in idx]
        bt = [bin_trues[i] for i in idx]
        ip = [int_preds[i] for i in idx]
        it = [int_trues[i] for i in idx]
        fb = _binary_f1(bp, bt)
        fc = _categorical_f1(bt, ip, it)
        return fb ** _ALPHA * fc ** (1 - _ALPHA)

    post_idx = [i for i, m in enumerate(modes) if m == "post"]
    comm_idx = [i for i, m in enumerate(modes) if m == "comment"]

    print(f"\n=== {split_name} ===")
    print(f"val_f1:        {val_f1:.4f}")
    print(f"f1_binary:     {f1_bin:.4f}")
    print(f"f1_categorical:{f1_cat:.4f}")
    print(f"f1_posts:      {mode_val_f1(post_idx):.4f}")
    print(f"f1_comments:   {mode_val_f1(comm_idx):.4f}")
    return val_f1


def main():
    t0 = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)

    all_train_rows = json.load(open("cache/dataset-train.json"))
    test_rows = json.load(open("cache/dataset-test.json"))

    # split train → train / val (no test leakage)
    random.shuffle(all_train_rows)
    n_val = int(len(all_train_rows) * VAL_FRAC)
    val_rows = all_train_rows[:n_val]
    train_rows = all_train_rows[n_val:]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)} (held out)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = BotDataset(train_rows, tokenizer)
    val_ds   = BotDataset(val_rows, tokenizer)
    test_ds  = BotDataset(test_rows, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BertModerator().to(device)
    bin_w = class_weights(train_rows, BINARY_LABELS, "label_binary").to(device)
    int_w = class_weights(train_rows, INTENT_LABELS, "label_intent").to(device)
    bin_loss_fn = nn.CrossEntropyLoss(weight=bin_w)
    int_loss_fn = nn.CrossEntropyLoss(weight=int_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            bl = batch["binary_label"].to(device)
            il = batch["intent_label"].to(device)
            bin_logits, int_logits = model(ids, mask)
            loss = bin_loss_fn(bin_logits, bl) + int_loss_fn(int_logits, il)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        val_f1 = evaluate(model, val_loader, device, f"VAL epoch {epoch+1}")
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    print(f"\nBest val_f1: {best_val_f1:.4f} — loading best checkpoint for test eval")
    model.load_state_dict(best_state)
    torch.save(best_state, "bert_best.pt")
    print("Saved bert_best.pt")
    evaluate(model, test_loader, device, "TEST (dataset-test.json)")
    print(f"\ntotal_seconds: {time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
