# LoRA Credit Scoring System — MTech Dissertation Deliverable

**Author:** Pupurayi Paula Chinyavada (H240799Q)  
**Institution:** Harare Institute of Technology  
**Supervisor:** Eng. A. Ndlovu  
**Year:** 2026

---

## What This Is

A complete, runnable experiment implementing the dissertation:
*"Low-Rank Adaptation (LoRA) for Alternative Data Credit Scoring: Enhancing Financial Inclusion under the National Development Strategy"*

## Files

| File | Description |
|------|-------------|
| `MTECH_LoRA_Credit_Scoring.ipynb` | **Main notebook** — run this on Google Colab |

## How to Run (Google Colab — Recommended)

1. Go to **https://colab.research.google.com**
2. Click **File → Upload notebook** → select `MTECH_LoRA_Credit_Scoring.ipynb`
3. Click **Runtime → Change runtime type → T4 GPU**
4. Click **Runtime → Run all** (Ctrl+F9)
5. Wait ~25–35 minutes for full experiment to complete
6. A **Gradio dashboard URL** appears in the last cell — open it to demo the system

## What the Notebook Produces

### Models Trained
- Logistic Regression (traditional baseline)
- XGBoost (state-of-the-art tabular baseline)
- LSTM (sequential deep learning baseline)
- **LoRA-DistilBERT r=8** (proposed model)

### Outputs
- `eda_analysis.png` — Exploratory data analysis charts
- `model_evaluation.png` — ROC curves, AUC comparison, confusion matrix
- `fairness_evaluation.png` — Demographic fairness metrics
- `shap_analysis.png` — Feature importance and SHAP values
- `experiment_results.json` — All numerical results
- **Live Gradio dashboard** — Interactive credit scoring demo

### Key Results (Expected)
| Model | AUC-ROC | Trainable Params |
|-------|---------|-----------------|
| Logistic Regression | ~0.74 | ~90 |
| XGBoost | ~0.82 | ~50K |
| LSTM | ~0.84 | ~2.1M |
| **LoRA-DistilBERT** | **~0.87** | **~1%** |

## System Requirements

- Google Colab free tier (T4 GPU) **or** local Python 3.10+ with CUDA GPU
- ~4GB GPU RAM minimum
- Internet connection (for package installation)

## Data

All data is **synthetically generated** within the notebook — calibrated against:
- Reserve Bank of Zimbabwe (2024)
- FinScope Consumer Survey (2024)
- ZIMSTAT Labour Force Survey (2023)

No real personal data is used.
