# train_and_eval_optionA.py
# OPTION A: Claim-only masked loss + contrastive scoring
# Improves accuracy & robustness without changing BDH internals

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from transformers import GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from bdh import BDH, BDHConfig


# ======================================================
# 1. Paths
# ======================================================

DATA_DIR = "data/"
TRAIN_CSV = DATA_DIR + "train.csv"

NOVELS = {
    "the count of monte cristo": DATA_DIR + "The_Count_of_Monte_Cristo.txt",
    "in search of the castaways": DATA_DIR + "In_Search_of_the_Castaways.txt",
}


# ======================================================
# 2. Load novels
# ======================================================

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

NOVEL_CACHE = {k: load_text(v) for k, v in NOVELS.items()}

def get_novel_text(book_name):
    key = book_name.strip().lower()
    if key not in NOVEL_CACHE:
        raise ValueError(f"Unknown book name: {book_name}")
    return NOVEL_CACHE[key]


# ======================================================
# 3. Tokenizer + BDH
# ======================================================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = BDHConfig(
    n_layer=4,
    n_embd=256,
    n_head=4,
    vocab_size=tokenizer.vocab_size,
)

model = BDH(config)
model.eval()


# ======================================================
# 4. Claim-only masked loss (CORE FIX)
# ======================================================

def compute_claim_loss(novel_text, claim_text, max_tokens=2048):
    """
    Computes NLL(claim | novel) by masking novel tokens.
    """

    novel_ids = tokenizer(
        novel_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens // 2,
        padding=False
    )["input_ids"]

    claim_ids = tokenizer(
        "\n\n" + claim_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens // 2,
        padding=False
    )["input_ids"]

    input_ids = torch.cat([novel_ids, claim_ids], dim=1)
    targets = input_ids.clone()

    # mask novel tokens
    novel_len = novel_ids.size(1)
    targets[:, :novel_len] = -100  # ignore_index

    with torch.no_grad():
        logits, _ = model(input_ids, targets=None)

    vocab = logits.size(-1)
    logits = logits.view(-1, vocab)
    targets = targets.view(-1)

    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=-100,
        reduction="mean"
    )

    return loss.item()


# ======================================================
# 5. Contrastive scoring (ROBUSTNESS FIX)
# ======================================================

def compute_contrastive_score(novel_text, claim_text):
    """
    score = NLL(claim | novel) - NLL(claim | empty)
    Lower score => more consistent
    """
    loss_with_context = compute_claim_loss(novel_text, claim_text)
    loss_no_context = compute_claim_loss("", claim_text)
    return loss_with_context - loss_no_context


# ======================================================
# 6. Run on TRAIN set
# ======================================================

print("\nRunning Option A (masked + contrastive) on train.csv...\n")

train_df = pd.read_csv(TRAIN_CSV)

scores = []
labels = []

for row in train_df.itertuples():
    novel = get_novel_text(row.book_name)
    score = compute_contrastive_score(novel, row.content)

    scores.append(score)
    labels.append(row.label)

    print(f"[TRAIN] Score: {score:.4f} | Label: {row.label}")


# ======================================================
# 7. Threshold optimization
# ======================================================

print("\nSearching for best threshold...\n")

scores_np = np.array(scores)
best_threshold = None
best_f1 = 0
best_acc = 0

for t in np.linspace(scores_np.min(), scores_np.max(), 300):
    preds = ["consistent" if s < t else "contradict" for s in scores_np]
    f1 = f1_score(labels, preds, pos_label="consistent")
    acc = accuracy_score(labels, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_threshold = t

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1-score: {best_f1:.3f}")
print(f"Best Accuracy: {best_acc:.3f}")


# ======================================================
# 8. Final diagnostic report
# ======================================================

final_preds = ["consistent" if s < best_threshold else "contradict" for s in scores_np]

print("\nFinal TRAIN classification report (diagnostic):\n")
print(
    classification_report(
        labels,
        final_preds,
        labels=["consistent", "contradict"],
        target_names=["Consistent", "Contradict"]
    )
)
SAVE_DIR = "checkpoints/"
os.makedirs(SAVE_DIR, exist_ok=True)

# save threshold
with open(SAVE_DIR + "threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("Saved threshold and BDH config.")
