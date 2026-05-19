# Causal-ML-Uplift-Engine-for-Digital-Marketing


This repository implements a production-grade **Double Machine Learning (DML)** pipeline to calculate **Conditional Average Treatment Effects (CATE)** using observational marketing data. 

Instead of traditional predictive tracking (which captures simple correlations), this system determines the exact **incremental revenue uplift** generated exclusively by advertising intervention.

## 📊 Dataset Reference
The pipeline runs natively on the **[Kaggle Causal Digital Marketing Campaign Dataset](https://www.kaggle.com/datasets/rahuljangir78/causal-digital-marketing-campaign-dataset)**. It isolates target treatment variables (`ad_exposed`) and handles multi-channel confounders natively.

## 🏗️ Core Methodology
To accurately estimate causality without confounding bias, this framework deploys **Orthogonal/Double Machine Learning**:
1. **Nuisance Model T**: Predicts propensity score of advertisement exposure using an `LGBMClassifier`.
2. **Nuisance Model Y**: Predicts baseline expected customer revenue outcomes via `LGBMRegressor`.
3. **Causal Cross-Fitting**: Minimizes residual bias to accurately compute true individual treatment variance.

## 🚀 How to Execute The System

### 1. Installation
Clone the repository and install locked constraints:
```bash
git clone https://github.com
cd causal-marketing-uplift
pip install -r requirements.txt
```

### 2. Prepare Data
Download the data file from Kaggle and place it in the designated folder structure:
`data/causal_digital_marketing_campaign_dataset.csv`

### 3. Run Production Pipeline
```bash
python main.py
```
