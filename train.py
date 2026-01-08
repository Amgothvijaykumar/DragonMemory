# train.py
# Track B — BDH loss-based narrative consistency evaluation

import torch
import pandas as pd

from transformers import GPT2Tokenizer
from bdh import BDH, BDHConfig


# ======================================================
# 1. Paths & dataset files (MATCH YOUR CSVs)
# ======================================================

DATA_DIR = "data/"

TRAIN_CSV = DATA_DIR + "train.csv"
TEST_CSV  = DATA_DIR + "test.csv"

NOVELS = {
    "the count of monte cristo": DATA_DIR + "The_Count_of_Monte_Cristo.txt",
    "in search of the castaways": DATA_DIR + "In_Search_of_the_Castaways.txt",
}


# ======================================================
# 2. Utilities
# ======================================================

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


NOVEL_CACHE = {
    key: load_text(path)
    for key, path in NOVELS.items()
}


def get_novel_text(book_name):
    key = book_name.strip().lower()
    if key not in NOVEL_CACHE:
        raise ValueError(f"Unknown book name in CSV: {book_name}")
    return NOVEL_CACHE[key]


# ======================================================
# 3. Tokenizer & BDH initialization (CORRECT API)
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
# 4. Core BDH scoring logic (THIS IS THE KEY)
# ======================================================

def compute_bdh_loss(context_text, max_tokens=2048):
    """
    Runs BDH in teacher-forcing mode and returns scalar loss.
    Lower loss => more compatible with narrative.
    """
    tokens = tokenizer(
        context_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        padding=False
    )["input_ids"]

    with torch.no_grad():
        out = model(tokens, tokens)

    # BDH may return loss or (loss, loss)
    if isinstance(out, tuple):
        loss = out[0]
    else:
        loss = out

    return loss.mean().item()



# ======================================================
# 5. TRAIN.CSV — threshold calibration (NOT training)
# ======================================================

print("Running BDH analysis on train.csv...")

train_df = pd.read_csv(TRAIN_CSV)

scores = []
labels = []

for row in train_df.itertuples():
    novel = get_novel_text(row.book_name)
    context = novel + "\n\n" + row.content

    loss = compute_bdh_loss(context)
    scores.append(loss)
    labels.append(row.label)

    print(f"[TRAIN] Loss: {loss:.4f} | Label: {row.label}")


# Simple threshold: midpoint between class means
consistent_losses = [s for s, l in zip(scores, labels) if l == "consistent"]
contradict_losses = [s for s, l in zip(scores, labels) if l == "contradict"]

THRESHOLD = (sum(consistent_losses)/len(consistent_losses) +
             sum(contradict_losses)/len(contradict_losses)) / 2

print(f"\nChosen loss threshold: {THRESHOLD:.4f}")


# ======================================================
# 6. TEST.CSV — final inference
# ======================================================

print("\nRunning BDH inference on test.csv...")

test_df = pd.read_csv(TEST_CSV)
predictions = []

for row in test_df.itertuples():
    novel = get_novel_text(row.book_name)
    context = novel + "\n\n" + row.content

    loss = compute_bdh_loss(context)
    pred = "consistent" if loss < THRESHOLD else "contradict"
    predictions.append(pred)


# ======================================================
# 7. Save output
# ======================================================

output = pd.DataFrame({
    "id": test_df.id,
    "prediction": predictions
})

output.to_csv("results.csv", index=False)

print("\nSaved results.csv")
