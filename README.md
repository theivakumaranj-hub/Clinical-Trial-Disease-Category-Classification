
# 🏥 Clinical Trial Disease Category Classification Using NLP and Machine Learning

![Alternative Text Label](ui.png)

## Domain & System Scope
* **Domain:** Healthcare Analytics, Medical NLP & Clinical Trial Intelligence Systems.
* **Framework Type:** Supervised Machine Learning Text Classification.
* **Primary Objective:** Transform unstructured clinical trial summaries into a structured, predictable format to automatically categorize large-scale medical research portfolios.

## Problem Statement
* The healthcare industry generates massive volumes of clinical trial data, including medical summaries, eligibility criteria, and disease-related information.
* This textual data is highly unstructured and difficult to analyze directly for extracting meaningful medical insights.
* This system solves the bottleneck by applying Natural Language Processing (NLP) and Machine Learning techniques to automatically predict and classify clinical trial summaries into appropriate disease categories using the `brief_summary` text.

---

## Technical Pipeline Engineering

### 1. Natural Language Processing (NLP) Preprocessing
* **Regex Noise Extraction:** Cleans raw text fields by stripping out special characters, numbers, and punctuation arrays.
* **Tokenization & Case Standardization:** Down-cases the text data globally and divides sentences into distinct word tokens.
* **Stop Word Removal:** Filters out high-frequency, low-meaning functional English syntax terms using standard NLTK dictionaries.
* **Morphological Lemmatization:** Converts inflected verbs and pluralized nouns back to their base dictionary roots using WordNet engines to preserve core semantic definitions.

### 2. Advanced Feature Engineering
* **N-Gram Token Range Mapping:** Captures sequential pairs of words (bigrams like "breast cancer") alongside single words (unigrams) to retain critical clinical medical context.
* **Sublinear Term Frequency Scaling:** Applies a logarithmic count scaling formula ($1 + \log(\text{tf})$) to dampen repetitive, generic protocol words while highlighting rare diagnostic markers.
* **Document Frequency Filtering:** Incorporates a strict `min_df=2` cut-off threshold to automatically purge unique spelling errors, patient IDs, or non-recurring character sequences from the vocabulary.
* **High-Dimensional Vocabulary Expansion:** Expands feature capacity up to a 25,000 text-pattern matrix to handle complex datasets.

### 3. Machine Learning Classification Model
* **Algorithm Choice:** Implements a fast, highly memory-efficient Logistic Regression classifier optimized for high-dimensional sparse text vectors.
* **Optimization Solver:** Utilizes the `saga` optimization solver engine, engineered specifically for fast, stable convergence on large data files (160MB dataset).
* **Decision Boundary Tuning:** Tightens the inverse regularization penalty constraint up to `C=5.0` to maximize prediction accuracy across target boundaries.

---

## System Performance Report

### Verified Classification Evaluation Metrics
* The predictive pipeline achieved the following highly balanced validation matrix on unseen testing subsets:

| Metric Criteria | Final Achieved System Score |
| :--- | :--- |
| **Global Classification Accuracy** | **95.00%** |
| **Weighted Precision Score** | **95.07%** |
| **Weighted Recall Sensitivity** | **95.00%** |
| **Harmonic F1-Score Value** | **94.99%** |

---

## Interactive Application Interface Layout

### Tab 1 — Real-Time Inference System
* Features a text area input box allowing users to paste unstructured medical trial summaries.
* Cleans, normalizes, and vectorizes the text strings instantly using the pre-trained token pipeline configurations.
* Runs real-time model inference and outputs the predicted disease category instantly.

### Tab 2 — Vertical 4-Chart Exploratory Data Analysis (EDA) Panel
* Displays analytical data visual panels in a cleanly stacked vertical sequence for optimal readability:
  1. **Top Verified Disease Categories Chart:** Horizontal bar chart profiling class distribution patterns across the data.
  2. **Most Frequent Medical Terms Chart:** Word frequency plot rendering the top 20 foundational words discovered across summaries.
  3. **Clinical Trial Phase Distribution Chart:** Tracks clinical volume milestones across experimental phases (Phase 1 to Phase 4).
  4. **Distribution by Study Design Type Chart:** Compares the structural balance between Interventional and Observational layouts.

---

## Local Installation & Operation Guide

### 1. Environmental Dependency Installation
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn streamlit joblib
