# ML BlackSwan: V-Shape Recovery Classification in S&P 500 Corrections

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)](https://lightgbm.readthedocs.io/)

A machine learning system that classifies post-correction recovery patterns in the S&P 500 as **V-shape** (rapid recovery >5% in 60 trading days) or **Non-V** (structural decline or slow recovery), using a combination of technical and macroeconomic features.

---

## Motivation

When markets correct significantly, investors face a critical question: is this a temporary dip followed by rapid recovery, or the beginning of a prolonged structural decline? This project addresses that question through an iterative ML pipeline, motivated by real market events including the Liberation Day tariffs correction of 2025.

---

## Results

| Model | AUC | Accuracy |
|---|---|---|
| Naive Bayes | 0.778 | 0.735 |
| LightGBM | 0.754 | 0.748 |
| Random Forest | 0.675 | 0.669 |
| **Voting Ensemble V5** | **0.804** | **0.742** |

- **Test set:** 151 weeks across 3 unseen crises (COVID-19, Fed Rate Hikes 2022, Liberation Day 2025)
- **Benchmark comparison:** +0.264 AUC over best non-ML signal (Momentum Signal, AUC=0.540)
- **Statistical significance:** DeLong, McNemar and Permutation tests all p<0.001

---

## Model Evolution

| Version | Frequency | Task | Best Model | Metric |
|---|---|---|---|---|
| V1 | Daily | 3 classes (V/U/L) | KNN | Acc 0.522 |
| V2 | Daily | 3 dynamic classes | KNN | Acc 0.495 |
| V3 | Daily | Binary | Naive Bayes | AUC 0.705 |
| V4 | Weekly | Binary | LightGBM | AUC 0.764 |
| V5 | Weekly | Binary + macro FRED | Voting Ensemble | AUC 0.804 |

---

## Features (18 total)

**Technical (12):** Return_1w, Return_4w, Dist_MA50, Dist_MA200, Dist_Local_Min, VIX, VIX_Ratio, VIX_Trend_20d, Volume_Ratio, RSI, SP500_Trend_20d, Phase

**Macroeconomic (6):** Yield_Curve, Yield_Curve_Change, Jobless_Ratio, Credit_Spread, Dollar_Change, Fed_Rate

Key finding: **Fed_Rate is the strongest single predictor** in both Random Forest and LightGBM, confirming that monetary policy context dominates recovery type classification.

---

## Project Structure

```
ML BlackSwan/
├── V1/                          # Daily classification, 3 classes
├── V2/                          # Dynamic labels
├── V3/                          # Binary classification, daily
├── V4/                          # Weekly aggregation
├── V5/
│   ├── v5_data.ipynb            # Data pipeline
│   ├── v5_modeling.ipynb        # Model training
│   ├── dashboard_v5.ipynb       # Weekly live dashboard
│   └── models/
│       ├── v5_ensemble.pkl      # Trained Voting Ensemble
│       ├── v5_scaler.pkl        # StandardScaler
│       └── v5_model_meta.json   # Model metadata
└── Paper/
    ├── sensitivity_analysis.ipynb
    ├── expanding_window.ipynb
    ├── benchmarks.ipynb
    ├── statistical_tests.ipynb
    ├── economic_significance_v6.ipynb
    ├── data/                    # CSV results
    └── plots/                   # Generated figures
```

---

## Weekly Dashboard

The model runs as a live weekly dashboard every Monday morning. It automatically downloads S&P 500, VIX and 4 FRED series, computes all 18 features and displays the current signal with P(V-shape) probability.

```python
# Run in V5/dashboard_v5.ipynb
# Restart & Run All every Monday morning
```

**Current signal logic:**
- P(V-shape) >= 0.62 → **V-SHAPE** signal (rapid recovery expected)
- P(V-shape) < 0.62 → **NON-V** signal (no rapid recovery expected)

---

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/ml-blackswan.git
cd ml-blackswan

# Create environment
conda create -n thesis_portfolio python=3.10
conda activate thesis_portfolio

# Install dependencies
pip install pandas numpy scikit-learn lightgbm yfinance fredapi matplotlib seaborn jupyter
```

Add your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html) in the second cell of `dashboard_v5.ipynb`.

---

## Validation Methodology

- **Sensitivity Analysis:** 9 parameter combinations (threshold 3/5/7% × forward 45/60/90 days), AUC std=0.034
- **Expanding Window:** 7 rounds, correlation between training events and AUC = 0.795
- **Bootstrap CI:** 1000 iterations per round
- **Statistical Tests:** DeLong, McNemar, Permutation (all p<0.001 vs benchmarks)

**Key finding:** Model requires a minimum of 7 diverse crisis episodes in training for reliable out-of-sample performance.

---

## Data Sources

| Data | Source | Series |
|---|---|---|
| S&P 500 price & volume | Yahoo Finance | ^GSPC |
| VIX | Yahoo Finance | ^VIX |
| Yield Curve | FRED | T10Y2Y |
| Jobless Claims | FRED | ICSA |
| Credit Spread | FRED | BAMLH0A0HYM2 |
| Fed Funds Rate | FRED | FEDFUNDS |

---

## Technical Report

A full technical report documenting the methodology, results and limitations is available in the repository as `report_vshape.docx`.

---

## Limitations

- 10 historical crisis events is a small dataset for ML
- Model validated exclusively on S&P 500; generalization to other indices untested
- Dollar_Change feature imputed with 0 due to FRED data availability issues
- Minimum 7 diverse crises required for reliable performance

---

## Author

**Baciu Teodor Bogdan** — April 2026

*Built as an independent ML research project alongside a Udemy ML course.*

---

## License

MIT License
