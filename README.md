# 📊 EdTech Lead Conversion Analysis — CRM Intelligence

A data science project that combines Exploratory Data Analysis (EDA) and Machine Learning to predict lead conversion probability for an EdTech company, enabling smarter CRM prioritization.

---

## 🎯 Business Context

In a CRM pipeline, leads fall into two categories:

- **Closed leads** — have a known outcome (converted or not), used for training
- **Open leads** — outcome still unknown, the real target for prediction

The goal is to train a classification model on closed leads so it generalizes well to open leads, assigning each a **conversion probability score** that the sales team can act on:

- **High probability** → prioritize for follow-up
- **Low probability** → deprioritize or discard

---

## 📁 Project Structure

```
.
├── analise_leads.ipynb          # Main analysis notebook
├── datasets/
│   ├── leads.csv                # Raw dataset (9,240 leads, 37 features)
│   └── leads_cleaned.csv        # Cleaned dataset (output)
└── preprocessor_dataset_leads.pkl  # Saved preprocessing pipeline
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | `leads.csv` |
| Total records | 9,240 leads |
| Features | 37 columns |
| Target variable | `Converted` (binary: 0 or 1) |
| Class distribution | ~61.5% not converted / ~38.5% converted |

### Key features include

- **Lead metadata**: `Lead Origin`, `Lead Source`, `Last Activity`, `Last Notable Activity`
- **Behavioral signals**: `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit`
- **Asymmetrique scores**: `Asymmetrique Activity Score`, `Asymmetrique Profile Score`
- **Demographic/profile**: `Specialization`, `What is your current occupation`, `City`
- **Marketing channels**: `Search`, `Magazine`, `Digital Advertisement`, `Through Recommendations`

---

## 🔬 Analysis Pipeline

### 1. Data Loading & Inspection

Initial load and exploration using `df.info()`, `.head()`, and `.tail()` to understand data shape and types.

### 2. Feature Engineering & Data Cleaning

- Dropped irrelevant ID columns (`Prospect ID`, `Lead Number`)
- Removed columns with a single unique value (zero variance)
- Removed categorical columns with more than 25% missing/`'Select'` values
- Standardized inconsistent category labels (e.g., `'google'` → `'Google'`)
- Converted Yes/No boolean columns to binary (1/0)
- Dropped rows with remaining missing values in categorical and numeric columns

### 3. Exploratory Data Analysis (EDA)

- **Hit ratio / Conversion rate**: bar chart of the overall conversion distribution
- **Correlation heatmap**: of all numeric features vs. `Converted`
- **Box plots**: `TotalVisits`, `Total Time Spent on Website`, and `Page Views Per Visit` segmented by conversion outcome
- **Chi-squared independence tests**: assessed statistical significance of the relationship between `Converted` and:
  - `Lead Source`
  - `Lead Origin`
  - `Last Notable Activity`

### 4. Data Preparation

- Feature/target split (`X`, `y`)
- Preprocessing pipeline via `ColumnTransformer`:
  - Numeric features → `StandardScaler`
  - Categorical features → `OneHotEncoder` (with `handle_unknown='ignore'`)
- 80/20 train-test split (random state: 51)

### 5. Model Training

A **Bagging Classifier** with **Logistic Regression** as the base estimator:

```python
BaggingClassifier(
    estimator=LogisticRegression(),
    n_estimators=10,
    random_state=51
)
```

Bagging reduces variance by training multiple models on bootstrapped samples and averaging their predictions.

### 6. Model Evaluation

| Metric | Description |
|---|---|
| Accuracy | Overall correctness |
| Precision | Of predicted conversions, how many were real |
| Recall | Of actual conversions, how many were caught |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Visual breakdown of TP, TN, FP, FN |

### 7. Feature Importance

Computed by averaging the absolute coefficients across all 10 Logistic Regression estimators, then visualized as a horizontal bar chart — showing which features most influence conversion prediction.

### 8. Conversion Probability Scoring

`predict_proba()` outputs a probability score per lead. This is the core CRM utility: rather than a binary prediction, sales teams get a ranked list by conversion likelihood.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Statistics | `scipy.stats` (chi-squared test) |
| Machine learning | `scikit-learn` |
| Model persistence | `joblib` |

---

## ▶️ How to Run

1. Clone the repository and navigate to the project folder.

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy joblib
```

3. Make sure `leads.csv` is placed inside a `datasets/` folder at the project root.

4. Open and run the notebook:

```bash
jupyter notebook analise_leads.ipynb
```

5. The cleaned dataset and preprocessor will be saved automatically after execution.

---

## 💡 CRM Application

The conversion probability score produced by the model can be integrated directly into a CRM workflow:

```
Lead Score ≥ 0.75  →  High priority — assign to sales rep immediately
Lead Score 0.40–0.74  →  Medium priority — nurture campaign
Lead Score < 0.40  →  Low priority — automated follow-up or discard
```

This approach allows a sales team to focus human effort where it is most likely to generate revenue.

---

## 📌 Notes

- The model is trained only on closed leads; open leads are scored using `predict_proba()` without retraining.
- The saved `preprocessor_dataset_leads.pkl` must be used for any future inference to ensure consistent feature transformation.
- Threshold values for CRM bucketing should be adjusted based on business-specific cost of false positives vs. false negatives.