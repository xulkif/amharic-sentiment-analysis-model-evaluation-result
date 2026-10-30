
from pathlib import Path
import json
import platform

import datasets
import sklearn
import torch
import transformers
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==================================================
# 1. EXPERIMENT CONFIGURATION
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

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 60)
print("AMHARIC SENTIMENT CLASSIFIER EVALUATION")
print("=" * 60)

dataset = load_dataset(DATASET_NAME)
test_data = dataset[DATASET_SPLIT]

label_names = test_data.features[LABEL_COLUMN].names

print(f"\nDataset: {DATASET_NAME}")
print(f"Split: {DATASET_SPLIT}")
print(f"Text column: {TEXT_COLUMN}")
print(f"Label column: {LABEL_COLUMN}")
print(f"Label order: {label_names}")
print(f"Original test examples: {len(test_data)}")

if not label_names:
    raise ValueError("No dataset class labels were found.")


def normalize_label(label):
    """Normalize label formatting for comparison."""
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


# ==================================================
# 3. LOAD PRETRAINED MODEL
# ==================================================

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device,
)

raw_id2label = classifier.model.config.id2label

model_id2label = {
    int(key): str(value)
    for key, value in raw_id2label.items()
}

print(f"\nModel: {MODEL_NAME}")
print(f"Device: {'CUDA GPU' if device == 0 else 'CPU'}")
print(f"Model label mapping: {model_id2label}")


# ==================================================
# 4. VALIDATE LABEL MAPPING
# ==================================================

def resolve_prediction_id(predicted_label):
    """
    Convert a model prediction into a dataset class ID.

    Generic labels such as LABEL_0 are accepted only
    when the model configuration identifies their
    corresponding semantic labels.
    """
    normalized = normalize_label(predicted_label)

    # The predicted label already matches the dataset.
    if normalized in dataset_label_to_id:
        return dataset_label_to_id[normalized]

    # Resolve generic model outputs using model config.
    if normalized.startswith("LABEL_"):
        suffix = normalized.removeprefix("LABEL_")

        if not suffix.isdigit():
            raise ValueError(
                f"Unrecognized model label: {predicted_label}"
            )

        model_class_id = int(suffix)
        configured_label = model_id2label.get(model_class_id)

        if configured_label is None:
            raise ValueError(
                f"Model class {model_class_id} is missing "
                "from the model's id2label configuration."
            )

        configured_normalized = normalize_label(configured_label)

        if configured_normalized.startswith("LABEL_"):
            raise ValueError(
                f"Cannot verify the semantic meaning of "
                f"{predicted_label}. The model configuration "
                f"also uses the generic label {configured_label!r}. "
                "Verify the model's label mapping from its "
                "training documentation before proceeding."
            )

        if configured_normalized not in dataset_label_to_id:
            raise ValueError(
                f"Model label {configured_label!r} does not "
                f"match dataset labels {label_names}."
            )

        return dataset_label_to_id[configured_normalized]

    raise ValueError(
        f"Cannot map prediction {predicted_label!r} "
        f"to dataset labels {label_names}."
    )


# Validate the model's configured classes before evaluation.
for class_id, class_name in model_id2label.items():
    normalized = normalize_label(class_name)

    if normalized.startswith("LABEL_"):
        continue

    if normalized not in dataset_label_to_id:
        raise ValueError(
            f"Model class {class_id} is {class_name!r}, "
            f"which does not match dataset labels {label_names}."
        )

# Resolve every configured class where possible.
for class_id, class_name in model_id2label.items():
    try:
        resolved_id = resolve_prediction_id(class_name)
        print(
            f"Verified model class {class_id}: "
            f"{class_name} -> {label_names[resolved_id]}"
        )
    except ValueError:
        if normalize_label(class_name).startswith("LABEL_"):
            # Generic mappings may be unused if actual
            # predictions return explicit sentiment names.
            continue
        raise


# ==================================================
# 5. VALIDATE TEST EXAMPLES
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
# 6. GENERATE PREDICTIONS
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

print("Predictions completed.")


# ==================================================
# 7. CALCULATE EVALUATION METRICS
# ==================================================

class_ids = list(range(len(label_names)))

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=class_ids,
    target_names=label_names,
    output_dict=True,
    zero_division=0,
    digits=4,
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=class_ids,
)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=class_ids,
        target_names=label_names,
        zero_division=0,
        digits=4,
    )
)

print("Confusion matrix:")
print("Rows = actual labels; columns = predicted labels")
print(f"Label order: {label_names}")
print(matrix)

correct = sum(
    actual == predicted
    for actual, predicted in zip(true_labels, predicted_labels)
)

print(f"\nCorrect predictions: {correct}")
print(f"Incorrect predictions: {len(texts) - correct}")


# ==================================================
# 8. SAVE REPRODUCIBILITY METADATA AND METRICS
# ==================================================

config_hash = getattr(
    classifier.model.config,
    "_commit_hash",
    None,
)

evaluation_record = {
    "experiment": {
        "dataset": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "dataset_fingerprint": getattr(
            test_data, "_fingerprint", None
        ),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "dataset_label_order": label_names,
        "model": MODEL_NAME,
        "model_commit_hash": config_hash,
        "model_id2label": model_id2label,
        "batch_size": BATCH_SIZE,
        "truncation": True,
        "device": "cuda" if device == 0 else "cpu",
    },
    "data_summary": {
        "original_test_examples": len(test_data),
        "evaluated_examples": len(texts),
        "excluded_examples": len(excluded_rows),
        "excluded_rows": excluded_rows,
    },
    "metrics": {
        "accuracy": float(accuracy),
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {
            name: report[name]
            for name in label_names
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_label_order": label_names,
        "correct_predictions": int(correct),
        "incorrect_predictions": int(len(texts) - correct),
    },
    "software_versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "scikit_learn": sklearn.__version__,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(
        evaluation_record,
        file,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nMetrics saved to: {METRICS_PATH}")
print("\nEvaluation completed successfully.")
