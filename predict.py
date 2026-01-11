# predict.py
# Generates results.csv from test.csv using BDH threshold-based reasoning
# Output format: id, prediction (1 = consistent, 0 = inconsistent)

import pandas as pd
from transformers import GPT2Tokenizer

from bdh import BDH, BDHConfig
from train import (
    compute_contrastive_score,
    get_novel_text
)

# =========================
# CONFIG
# =========================

THRESHOLD = 0.0643   # <-- best threshold from training
DATA_DIR = "data/"
TEST_CSV = DATA_DIR + "test.csv"

# =========================
# Load tokenizer + BDH
# =========================

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

# =========================
# Run inference
# =========================

test_df = pd.read_csv(TEST_CSV)

rows = []

for row in test_df.itertuples():
    novel = get_novel_text(row.book_name)
    score = compute_contrastive_score(novel, row.content)

    pred = 1 if score < THRESHOLD else 0

    rows.append({
        "id": row.id,
        "prediction": pred
    })

results_df = pd.DataFrame(rows)
results_df.to_csv("results.csv", index=False)

print("results.csv generated successfully (id, prediction).")
