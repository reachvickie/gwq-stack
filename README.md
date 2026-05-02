# GWQ-Stack: Fault-Tolerant Neuro-Symbolic Ensemble for Global Water Quality Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Figshare-lightgrey.svg)](https://doi.org/10.6084/m9.figshare.27800394.v2)

> **GWQ-Stack** predicts the Canadian Council of Ministers of the Environment Water Quality Index (CCME-WQI) from eight physicochemical parameters across 2.82 million globally distributed surface water measurements. It combines four gradient boosting base learners with a neural MLP meta-learner, trained under a leakage-free chronological holdout, and includes a three-tier fault tolerance cascade for partial sensor failure.

---

## Table of Contents

- [Results at a Glance](#results-at-a-glance)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quickstart — Reproduce Main Results](#quickstart--reproduce-main-results)
- [Fault Tolerance Experiment](#fault-tolerance-experiment)
- [Architecture](#architecture)
- [License](#license)

---

## Results at a Glance

| Metric | Score | Evaluation set |
|--------|-------|----------------|
| R² | **0.9998** | Temporal holdout (post-2015, n = 704,249) |
| RMSE | **0.1577** | Temporal holdout |
| MAE | **0.0538** | Temporal holdout |

Validation strategy: strict pre-2015 / post-2015 chronological split + GroupKFold by monitoring station (no temporal or station-level leakage).

---

## Repository Structure

```
gwq-stack/
│
├── gwq.py                  # Full training pipeline (GroupKFold + temporal split + stacking)
├── experiment.py           # Fault tolerance experiment (k=1,2,3 sensor dropout)
│
├── model/
│   └── gwq_final_pipeline.pkl   # Serialised pipeline 
│
├── plots/                 
│   ├── Fig1_Loss_Curves.png
│   ├── Fig2_SHAP_Importance.png
│   ├── Fig3_Actual_vs_Predicted.png
│   ├── Fig4_Correlation_Heatmap.png
│   ├── Fig5_Feature_Distributions.png
│   ├── Fig6_Architecture.png
│   └── FigFT_Fault_Tolerance.png
│
├── requirements.txt        # Exact Python environment
├── LICENSE
└── README.md
```

---

## Requirements

Python **3.12** is required. All dependencies are pinned for exact reproducibility.

```
catboost==1.2.10
xgboost==2.1.1
lightgbm==4.3.0
scikit-learn==1.4.2
shap==0.45.0
pandas==2.2.2
numpy==2.0.2
matplotlib==3.10.0
seaborn==0.13.2
joblib==1.4.2
scipy==1.16.3
```

---

## Installation

**pip**

```bash
git clone https://github.com/reachvickie/gwq-stack.git
cd gwq-stack
pip install -r requirements.txt
```

**GPU note:** XGBoost and CatBoost will automatically use CUDA if an NVIDIA GPU is available. The pipeline was trained on a Google Colab T4 GPU. CPU-only execution is fully supported but training will be slower (~5–8× for large folds).

---

## Dataset

The paper uses the publicly available global surface water quality dataset:

> Karim, M. R. et al. *A comprehensive dataset of surface water quality spanning 1940–2023 for empirical and ML-adopted research.* **Scientific Data** 12, 391 (2025).  
> https://doi.org/10.6084/m9.figshare.27800394.v2

**Download steps:**

1. Go to https://doi.org/10.6084/m9.figshare.27800394.v2
2. Download `Combined_dataset.csv`
3. Place it at the path referenced in `gwq.py`:

Here is the link to download the Combined_dataset.csv : 
https://figshare.com/ndownloader/files/50757321
```python
FILEPATH = "/path/to/your/Combined_dataset.csv"
```

**Dataset summary:**

| Property | Value |
|---|---|
| Total records | 2,827,977 |
| Countries | USA, Canada, Ireland, England, China |
| Date range | 1940–2023 |
| Parameters | Ammonia, BOD, DO, Orthophosphate, pH, Temperature, Nitrogen, Nitrate |
| Target | CCME-WQI (pre-computed, 0–100 scale) |

---

## Quickstart — Reproduce Main Results

**Step 1 — Set your dataset path**

Open `gwq.py` and update line 12:

```python
FILEPATH = "/path/to/Combined_dataset.csv"
```

**Step 2 — Run the full training pipeline**

```bash
python gwq.py
```

This will:
1. Load and preprocess the dataset (label encoding, temporal split)
2. Apply the three-tier cascade imputation on train and test sets
3. Train four base learners (XGBoost, CatBoost, LightGBM, XGBoost-RF) using 5-fold GroupKFold
4. Train the MLP meta-learner (64→32) on out-of-fold predictions
5. Evaluate on the temporal holdout (post-2015)
6. Print R², RMSE, MAE
7. Save the serialised pipeline to `model/gwq_final_pipeline.pkl`
8. Generate all paper figures to `plots/`

**Expected output (temporal holdout):**

```
Train: 2,123,728 rows  |  Test: 704,249 rows
...
FINAL RESULTS (Temporal Holdout, post-2015)
R²   : 0.99980
RMSE : 0.15770
MAE  : 0.05380
Pipeline saved → model/gwq_final_pipeline.pkl
```

**Runtime:** approximately 30 -45 mins on a T4 GPU (Google Colab). 


## Fault Tolerance Experiment

To reproduce the sensor dropout experiment from Table 3 of the paper:

**Step 1 — Set your dataset path**

Open `experiment.py` and update:

```python
FILEPATH = "/path/to/Combined_dataset.csv"
PKL_PATH = "/path/to/model/gwq.pkl"
```

**Step 2 — Run the experiment**

```bash
python experiment.py
```

This will:
- Test k=1, k=2, k=3 simultaneous sensor failures
- At 10%, 30%, and 50% dropout rates
- Comparing No Imputation vs Tier-3 (global median) vs Tier-2 (temporal forward-fill)
- Print the full results table matching Table 3 in the paper
- Save three fault tolerance figures to `plots/`

**Expected output (excerpt):**

```
=== FINAL RESULTS TABLE ===
             Scenario  Drop  Baseline  No_Impute    Tier3    Tier2
        Ammonia (k=1)   0.1    0.15777    1.52187  1.52177  1.24283
        Ammonia (k=1)   0.3    0.15777    2.63457  2.63440  2.16814
...
NH3 + PO4 + BOD (k=3)   0.5    0.15777   10.74068  8.68778  6.07835
```

---

## Architecture

```
Raw sensor input (8 parameters + 3 categorical)
        │
        ▼
┌─────────────────────────────────────────┐
│     Three-Tier Fault Tolerance Cascade  │
│  Tier-2: temporal forward-fill          │
│  Tier-1: spatial waterbody-type median  │
│  Tier-3: global median fallback         │
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              Level 0 — Base Learners (GroupKFold × 5)    │
│                                                          │
│  XGBoost (GPU)   CatBoost (GPU)   LightGBM   XGB-RF (GPU)│
│                                                          │
│        Out-of-fold predictions accumulated               │
└──────────────────────────────────────────────────────────┘
        │
        ▼  StandardScaler
┌────────────────────────────────────┐
│   Level 1 — MLP Meta-Learner       │
│   Input(4) → Dense(64) → Dense(32) │
│   → Output(1)  [ReLU, Adam]        │
└────────────────────────────────────┘
        │
        ▼
   CCME-WQI prediction (0–100)
```

**Validation strategy:**

| Split | Records | Date range |
|---|---|---|
| Training | 2,123,728 | 1940 – Dec 2014 |
| Temporal holdout | 704,249 | Jan 2015 – 2023 |

GroupKFold within training: groups = monitoring station (`Area_encoded`). No station appears in both training and validation within any fold.



---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Vignesh A — vignesh.a2025@vitstudent.ac.in  
School of Computer Science and Engineering  
Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India

Issues and pull requests are welcome.
