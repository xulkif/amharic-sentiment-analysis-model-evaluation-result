# Error Analysis of Fine-Tuned Amharic Sentiment Classifier

An independent evaluation and qualitative error analysis of open-source Amharic sentiment classification models. This study evaluates [`rasyosef/bert-medium-amharic-finetuned-sentiment`](https://huggingface.co/rasyosef/bert-medium-amharic-finetuned-sentiment) on the public test split of [`rasyosef/amharic-sentiment`](https://huggingface.co/datasets/rasyosef/amharic-sentiment) (279 tweets).

Accompanying research paper: [Technical Report (`results/amharic_sentiment_error_analysis_final_review.docx`)](results/amharic_sentiment_error_analysis_final_review.docx).

---

## 📌 Key Findings

- **Test Accuracy**: **83.15%** (232 correct out of 279 test examples; 47 misclassifications).
- **Macro F1**: **0.8284**
  - **Negative**: Precision `0.8084`, Recall `0.9000`, F1 `0.8517` (Support: 150)
  - **Positive**: Precision `0.8661`, Recall `0.7519`, F1 `0.8050` (Support: 129)
- **Asymmetric Misses**: Positive recall is significantly lower (0.75) than negative recall (0.90). 32 of the 47 errors were positive tweets misclassified as negative, compared to only 15 negative tweets misclassified as positive.
- **3-Annotator Human Study on the 47 Errors**:
  - Independent review by 3 native Amharic speakers (allowing `positive`, `negative`, `neutral/mixed`, and `uncertain`).
  - **Fleiss' Kappa**: \(\kappa \approx 0.209\) (reflecting slight/fair agreement, highlighting that misclassifications are often linguistically or pragmatically ambiguous).
  - **Triage of Errors**:
    - **21 tweets (44.7%)**: Human majority agreed with the dataset label (genuine model errors).
    - **4 tweets (8.5%)**: Human majority agreed with the model prediction (candidates for label noise/re-examination: rows 29, 89, 144, 239).
    - **10 tweets (21.3%)**: Human majority selected `neutral/mixed` or `uncertain` (forcing binary polarity drops vital information).
    - **12 tweets (25.5%)**: Complete 3-way split with no majority (inherent subjectivity, sarcasm, or missing target).

---

## 📁 Repository Structure

```text
├── evaluate.py                  # Evaluates model on test split and exports metrics JSON
├── error_analysis.py            # Extracts misclassified tweets and generates error CSV
├── prepare_review_template.py   # Prepares UTF-8-BOM CSV template for multi-annotator review
├── compute_agreement.py         # Computes Fleiss' Kappa and agreement breakdown
├── requirements.txt             # Python dependencies
├── .gitignore                   # Standard Git exclusions for ML and OS files
├── README.md                    # Project documentation
└── results/
    ├── amharic_sentiment_error_analysis_final_review.docx  # Full research report
    ├── evaluation_metrics.json                             # Detailed scores & environment specs
    ├── error_analysis_summary.json                         # Misclassification metadata
    ├── error_analysis_misclassified.csv                    # 47 misclassified tweets with predictions
    └── human_review_template.csv                           # Annotation sheet for reviewers
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites & Virtual Environment

Python 3.10+ is recommended.

```bash
# Clone the repository
git clone https://github.com/<your-username>/amharic-sentiment-error-analysis.git
cd amharic-sentiment-error-analysis

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Model Evaluation

Runs inference across the 279 test tweets, validates label alignment, outputs classification metrics, and records hardware/software metadata:

```bash
python evaluate.py
```
*Outputs:* `results/evaluation_metrics.json`

### 3. Extract Errors for Analysis

Extracts the 47 misclassified examples with model confidence scores and true labels:

```bash
python error_analysis.py
```
*Outputs:* `results/error_analysis_misclassified.csv` and `results/error_analysis_summary.json`

### 4. Generate Human Review Sheet

Creates an Excel-ready UTF-8 with BOM template formatted for multi-rater annotation:

```bash
python prepare_review_template.py
```
*Outputs:* `results/human_review_template.csv`

---

## 🔬 Dataset & Model Details

- **Dataset**: [`rasyosef/amharic-sentiment`](https://huggingface.co/datasets/rasyosef/amharic-sentiment)
  - Train: 2,223 tweets | Validation: 279 tweets | Test: 279 tweets
  - Upstream source: [liyaSileshi/amharic-sentiment-analysis](https://github.com/liyaSileshi/amharic-sentiment-analysis)
- **Model**: [`rasyosef/bert-medium-amharic-finetuned-sentiment`](https://huggingface.co/rasyosef/bert-medium-amharic-finetuned-sentiment)
  - Pretrained medium Amharic BERT architecture (~40.5M parameters) fine-tuned for binary sentiment classification.

---

## 📖 Citation

If you use this work or findings in your research, please cite:

```bibtex
@techreport{zulkif2026amharic,
  title={Error Analysis of a Fine-Tuned Amharic Sentiment Classifier},
  author={Zulkif},
  year={2026},
  institution={Independent Technical Report},
  url={https://github.com/<your-username>/amharic-sentiment-error-analysis}
}
```

---

## 📄 License

This repository is available under the [MIT License](LICENSE).
