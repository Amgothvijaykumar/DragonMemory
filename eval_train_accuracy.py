# eval_train_accuracy.py
# Diagnostic script: computes TRAIN accuracy only
# Does NOT modify train.py or results.csv

import pandas as pd
from train import (
    compute_bdh_loss,
    get_novel_text,
    THRESHOLD,
    TRAIN_CSV
)

print("Evaluating TRAIN accuracy (diagnostic only)...")

train_df = pd.read_csv(TRAIN_CSV)

correct = 0
total = 0

for row in train_df.itertuples():
    novel = get_novel_text(row.book_name)
    context = novel + "\n\n" + row.content

    loss = compute_bdh_loss(context)
    pred = "consistent" if loss < THRESHOLD else "contradict"

    if pred == row.label:
        correct += 1
    total += 1

accuracy = correct / total

print(f"\nTrain accuracy: {accuracy:.2f}")
