
from pathlib import Path
import pandas as pd

# Paths
PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "error_analysis_misclassified.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "human_review_template.csv"
)

# Check that the source file exists.
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the source file: {INPUT_FILE}"
    )

# Protect an existing review file from accidental overwriting.
if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"The review template already exists: {OUTPUT_FILE}\n"
        "Rename or back up the existing file before creating "
        "a replacement."
    )

# Load the original error analysis.
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

if df.empty:
    raise ValueError("The source CSV contains no misclassified examples.")

required_columns = {
    "test_row_index",
    "text",
    "actual_sentiment",
    "predicted_sentiment",
    "prediction_confidence",
    "is_correct",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Source CSV is missing columns: {sorted(missing_columns)}"
    )

# Add blank fields for independent human annotation.
df["human_review_label"] = ""
df["label_agreement"] = ""
df["error_category"] = ""
df["review_notes"] = ""
df["reviewer_id"] = ""
df["review_status"] = "not_reviewed"

# Save as UTF-8 with BOM for better Amharic compatibility in Excel.
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("Human-review template created successfully.")
print(f"Examples to review: {len(df)}")
print(f"Output file: {OUTPUT_FILE}")
print("\nOriginal CSV was not modified.")
