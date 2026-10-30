
from pathlib import Path
import csv
import json

import torch
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score


# ==================================================
# 1. EXPERIMENT CONFIGURATION
# Keep consistent with evaluate.py
# ==================================================

DATASET_NAME = "rasyosef/amharic-sentiment"
DATASET_SPLIT = "test"
TEXT_COLUMN = "clean_tweet"
LABEL_COLUMN = "label"

MODEL_NAME = "rasyosef/bert-medium-amharic-finetuned-sentiment"
BATCH_SIZE = 8

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_PATH = RESULTS_DIR / "error_analysis_misclassified.csv"
SUMMARY_PATH = RESULTS_DIR / "error_analysis_summary.json"


# ==================================================
# 2. LOAD THE SAME DATASET AND MODEL
# ==================================================

dataset = load_dataset(DATASET_NAME)
test_data = dataset[DATASET_SPLIT]

label_names = test_data.features[LABEL_COLUMN].names

print("=" * 60)
print("AMHARIC SENTIMENT ERROR ANALYSIS")
print("=" * 60)
print(f"Dataset: {DATASET_NAME}")
print(f"Split: {DATASET_SPLIT}")
print(f"Label order: {label_names}")
print(f"Original test examples: {len(test_data)}")

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device,
)

model_id2label = {
    int(key): str(value)
    for key, value in classifier.model.config.id2label.items()
}

print(f"Model: {MODEL_NAME}")
print(f"Model label mapping: {model_id2label}")
print(f"Device: {'CUDA GPU' if device == 0 else 'CPU'}")


# ==================================================
# 3. VALIDATE AND MAP LABELS
# Same mapping approach as evaluate.py
# ==================================================

def normalize_label(label):
    return (
        str(label)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )


dataset_label_to_id = {
    normalize_label(name): index
    for index, name in enumerate(label_names)
}


def resolve_prediction_id(predicted_label):
    normalized = normalize_label(predicted_label)

    if normalized in dataset_label_to_id:
        return dataset_label_to_id[normalized]

    if normalized.startswith("LABEL_"):
        suffix = normalized.removeprefix("LABEL_")

        if not suffix.isdigit():
            raise ValueError(
                f"Unrecognized model label: {predicted_label}"
            )

        class_id = int(suffix)
        configured_label = model_id2label.get(class_id)

        if configured_label is None:
            raise ValueError(
                f"Model class {class_id} is missing from id2label."
            )

        configured_normalized = normalize_label(configured_label)

        if (
            configured_normalized.startswith("LABEL_")
            or configured_normalized not in dataset_label_to_id
        ):
            raise ValueError(
                f"Cannot verify the meaning of model label "
                f"{predicted_label!r} using model configuration "
                f"{model_id2label}."
            )

        return dataset_label_to_id[configured_normalized]

    raise ValueError(
        f"Cannot map model label {predicted_label!r} "
        f"to dataset labels {label_names}."
    )


# Validate the configured class mapping.
for class_id, class_name in model_id2label.items():
    normalized = normalize_label(class_name)

    if normalized.startswith("LABEL_"):
        continue

    if normalized not in dataset_label_to_id:
        raise ValueError(
            f"Model class {class_id}: {class_name!r} does not "
            f"match dataset labels {label_names}."
        )

print("\nVerified model classes:")
for class_id, class_name in model_id2label.items():
    normalized = normalize_label(class_name)

    if normalized.startswith("LABEL_"):
        continue

    print(
        f"  {class_id}: {class_name} -> "
        f"{label_names[dataset_label_to_id[normalized]]}"
    )


# ==================================================
# 4. PREPARE VALID TEST EXAMPLES
# Preserve original row indices
# ==================================================

texts = []
true_labels = []
original_indices = []
excluded_rows = []

for index, row in enumerate(test_data):
    text = row[TEXT_COLUMN]
    label = row[LABEL_COLUMN]

    if not isinstance(text, str) or not text.strip():
        excluded_rows.append({
            "test_row_index": index,
            "reason": "Missing or invalid text",
        })
        continue

    try:
        label_id = int(label)
    except (TypeError, ValueError):
        excluded_rows.append({
            "test_row_index": index,
            "reason": "Invalid label",
        })
        continue

    if not 0 <= label_id < len(label_names):
        excluded_rows.append({
            "test_row_index": index,
            "reason": "Label ID out of range",
        })
        continue

    texts.append(text.strip())
    true_labels.append(label_id)
    original_indices.append(index)

print(f"\nValid examples: {len(texts)}")
print(f"Excluded examples: {len(excluded_rows)}")

if not texts:
    raise ValueError("No valid test examples remain.")


# ==================================================
# 5. GENERATE PREDICTIONS
# ==================================================

print("\nGenerating predictions...")

results = classifier(
    texts,
    batch_size=BATCH_SIZE,
    truncation=True,
)

if len(results) != len(texts):
    raise RuntimeError(
        "Prediction count does not match valid test examples."
    )

predicted_labels = [
    resolve_prediction_id(result["label"])
    for result in results
]


# ==================================================
# 6. COLLECT MISCLASSIFIED EXAMPLES
# ==================================================

errors = []

for row_index, text, actual_id, predicted_id, result in zip(
    original_indices,
    texts,
    true_labels,
    predicted_labels,
    results,
):
    if actual_id == predicted_id:
        continue

    errors.append({
        "test_row_index": row_index,
        "text": text,
        "actual_sentiment": label_names[actual_id],
        "predicted_sentiment": label_names[predicted_id],
        "prediction_confidence": round(
            float(result["score"]), 6
        ),
        "is_correct": False,
    })


# ==================================================
# 7. SAVE MISCLASSIFIED EXAMPLES
# ==================================================

columns = [
    "test_row_index",
    "text",
    "actual_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "is_correct",
]

with open(
    ERRORS_PATH,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(errors)


# ==================================================
# 8. SAVE SUMMARY AND CHECK CONSISTENCY
# ==================================================

accuracy = accuracy_score(true_labels, predicted_labels)
correct_count = sum(
    actual == predicted
    for actual, predicted in zip(true_labels, predicted_labels)
)

summary = {
    "dataset": DATASET_NAME,
    "split": DATASET_SPLIT,
    "model": MODEL_NAME,
    "dataset_label_order": label_names,
    "model_id2label": model_id2label,
    "original_test_examples": len(test_data),
    "evaluated_examples": len(texts),
    "excluded_examples": len(excluded_rows),
    "correct_predictions": correct_count,
    "incorrect_predictions": len(errors),
    "accuracy": float(accuracy),
    "error_csv": str(ERRORS_PATH),
    "excluded_rows": excluded_rows,
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)


# ==================================================
# 9. PRINT RESULTS
# ==================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)
print(f"Original test examples: {len(test_data)}")
print(f"Valid examples evaluated: {len(texts)}")
print(f"Excluded examples: {len(excluded_rows)}")
print(f"Correct predictions: {correct_count}")
print(f"Misclassified examples: {len(errors)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print(f"\nMisclassified examples saved to:\n{ERRORS_PATH}")
print(f"\nSummary saved to:\n{SUMMARY_PATH}")

print("\nError analysis completed.")
