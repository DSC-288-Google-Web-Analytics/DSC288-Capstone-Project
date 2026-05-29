<a id="top"></a>

<div align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:A7C7E7,25:F4B6C2,50:F9E79F,75:B7E4C7,100:CDB4DB&height=190&section=header&text=Google%20Analytics%20Revenue%20Prediction&fontSize=32&fontColor=2F3E46&animation=fadeIn&fontAlignY=33&desc=Session-Level%20Revenue%20Forecasting%20%7C%20Gradient%20Boosting%20%7C%20Leakage-Safe%20Features&descAlignY=54&descSize=15"
    style="display: block; margin: 0 auto;"
  />
</div>

<div align="center">
  <h3><i>Predicting Customer Revenue from Google Analytics Session Data</i></h3>
  <h4>DSC 288R: Capstone Project</h4>

  <p>
    <strong>Pooja Panchal</strong> (Project Organization, Data Engineering, JSON Extraction, Report Write-Up)
    &nbsp;&bull;&nbsp;
    <strong>Jinxin Xiao</strong> (EDA, LightGBM Implementation)
    &nbsp;&bull;&nbsp;
    <strong>Justin Chanthabandith</strong> (Data Access, EDA, Modeling Pipeline, XGBoost Implementation)
  </p>

  <div>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
    <img src="https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white" />
    <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge" />
    <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  </div>
</div>

---

<p align="center">
  <a href="#1-introduction">Introduction</a> |
  <a href="#2-data-and-problem-structure">Data</a> |
  <a href="#3-methods">Methods</a> |
  <a href="#4-results">Results</a> |
  <a href="#5-discussion">Discussion</a> |
  <a href="#6-project-completion-summary">Project Summary</a> |
  <a href="#7-statement-of-collaboration">Collaboration</a> |
  <a href="#repository-structure">Structure</a> |
  <a href="#how-to-run">How to Run</a> |
  <a href="#references">References</a>
</p>

---

## 1. Introduction
[Back to Top](#top)

### Why This Project?

E-commerce companies collect large volumes of web analytics data that describe how visitors arrive at a site, what devices they use, where they are located, how they interact with each session, and whether those sessions lead to purchases. This project uses the **Google Analytics Customer Revenue Prediction** dataset to predict session-level customer revenue for the Google Merchandise Store.

The task is difficult because most sessions produce no revenue. This creates a **zero-inflated regression problem** where the model must learn patterns for a very small number of revenue-generating sessions while avoiding overprediction for the large majority of zero-revenue sessions.

### Research Question

> Can session-level Google Analytics features predict customer revenue, and do gradient boosting models outperform baseline, linear, random forest, and two-step modeling approaches on a highly zero-inflated target?

### Project Overview

| Aspect | Detail |
|---|---|
| **Problem Type** | Zero-inflated session-level revenue prediction |
| **Target Variable** | `totals.transactionRevenue` |
| **Modeling Target** | `log_revenue = log1p(transactionRevenue)` |
| **Primary Metric** | RMSE on log-transformed revenue |
| **Dataset** | Google Analytics Customer Revenue Prediction, Kaggle 2018 |
| **File Used** | `train_v2.csv` |
| **Data Scale** | 1,708,337 sessions and 1,323,730 unique visitors |
| **Positive Revenue Rate** | 1.0837% of sessions |
| **Train/Test Split** | Visitor-grouped 80/20 split using `fullVisitorId` |
| **Best Final Model** | LightGBM Regressor |
| **Best Test RMSE** | 1.5192 on log-transformed revenue |
| **LightGBM Test MAE** | 0.2667 on log-transformed revenue |
| **Main Challenge** | About 98.9% of sessions have zero revenue, while positive revenue is highly skewed |

---

## 2. Data and Problem Structure
[Back to Top](#top)

### Dataset

The project uses `train_v2.csv` from Kaggle's **Google Analytics Customer Revenue Prediction** competition. Each row represents one anonymized Google Merchandise Store session.

| Feature Group | Examples |
|---|---|
| Visitor identity | `fullVisitorId` |
| Session timing | `date`, `visitStartTime`, `visitNumber` |
| Traffic source | `channelGrouping`, `trafficSource.medium`, `trafficSource.source` |
| Device | `device.deviceCategory`, `device.browser`, `device.operatingSystem` |
| Geography | `geoNetwork.country` |
| Session activity | `totals.hits`, `totals.pageviews` |
| Target | `totals.transactionRevenue` |

The raw file is not flat. Several columns, including `device`, `geoNetwork`, `totals`, and `trafficSource`, are stored as nested JSON-like strings. The final notebook extracts only the fields needed for EDA and modeling, then drops the raw nested columns to reduce memory usage.

### Key Dataset Challenges

| Challenge | Why It Matters | How We Address It |
|---|---|---|
| **Zero inflation** | Only 1.0837% of sessions generate revenue | Compare zero baseline, mean baseline, direct regressors, and two-step models |
| **Revenue skew** | Positive revenue values vary across a wide range | Use `log1p(transactionRevenue)` for training and evaluation |
| **Nested JSON fields** | Important predictors are hidden inside string columns | Extract selected fields from `device`, `geoNetwork`, `totals`, and `trafficSource` |
| **High-cardinality categories** | Source, browser, and country can create sparse features | Keep the top 25 categories and group the rest as `Other` |
| **Visitor leakage risk** | The same visitor can appear in multiple sessions | Split by `fullVisitorId` so visitors do not overlap across train and test sets |
| **Visitor history leakage** | Future sessions can accidentally influence historical features | Build cumulative visitor features using only prior sessions |
| **Large dataset size** | The file has over 1.7 million rows | Use Polars for loading and feature preparation before converting to pandas for modeling |
| **Metric sensitivity** | MAE rewards predicting zero too often | Use RMSE on log revenue as the primary metric |

---

## 3. Methods
[Back to Top](#top)

### 3.1 Preprocessing Pipeline

The final notebook prepares the raw data through the following workflow:

1. **Data acquisition**  
   The notebook supports downloading the Kaggle dataset through the Kaggle API. It uses `train_v2.csv` as the main input file.

2. **Efficient loading with Polars**  
   The dataset is loaded with Polars to handle the full 1,708,337-row file more safely than a standard full pandas load.

3. **Nested field extraction**  
   The notebook extracts selected fields from the nested JSON-like columns:
   - `totals`: `transactionRevenue`, `hits`, `pageviews`, `bounces`, `newVisits`
   - `device`: `deviceCategory`, `browser`, `operatingSystem`
   - `geoNetwork`: `country`
   - `trafficSource`: `medium`, `source`

4. **Target engineering**  
   Missing revenue values are treated as zero. The pipeline creates:
   - `revenue`
   - `has_revenue`
   - `log_revenue`

5. **Date and time features**  
   The notebook converts date and timestamp fields into:
   - `date_parsed`
   - `visit_start_dt`
   - `year`
   - `month`
   - `day_of_week`
   - `hour`

6. **EDA-driven feature selection**  
   The final feature set keeps engagement, traffic, device, geography, timing, and visitor history features. It drops `totals.bounces` and `totals.newVisits` from modeling because their binary-null structure becomes weak after imputation, and `visitNumber` captures the returning-visitor signal more reliably.

7. **Leakage-safe visitor history features**  
   Sessions are sorted by visitor and time. Cumulative visitor features are created using only prior sessions for the same visitor.

8. **Visitor-level train/test split**  
   The notebook uses `GroupShuffleSplit` with `fullVisitorId` as the group variable. This prevents the same visitor from appearing in both training and test data.

### 3.2 Final Feature Set

| Feature Group | Final Features |
|---|---|
| Session behavior | `totals.hits`, `totals.pageviews`, `visitNumber` |
| Traffic source | `channelGrouping`, `trafficSource.medium`, `trafficSource.source` |
| Device | `device.deviceCategory`, `device.browser`, `device.operatingSystem` |
| Geography | `geoNetwork.country` |
| Timing | `month`, `day_of_week`, `hour` |
| Visitor history | `visitor_prior_session_count`, `visitor_prior_log_total_revenue`, `visitor_prior_avg_session_revenue`, `visitor_prior_max_session_revenue`, `visitor_prior_purchase_rate` |

High-cardinality categorical features are reduced before one-hot encoding:

| Feature | Treatment |
|---|---|
| `trafficSource.source` | Keep top 25 values, map all others to `Other` |
| `device.browser` | Keep top 25 values, map all others to `Other` |
| `geoNetwork.country` | Keep top 25 values, map all others to `Other` |

### 3.3 Train/Test Split and Preprocessing

The final modeling table contains **1,708,337 rows and 21 columns**.

| Split Detail | Value |
|---|---:|
| Train rows | 1,366,852 |
| Test rows | 341,485 |
| Train visitors | 1,058,984 |
| Test visitors | 264,746 |
| Visitor overlap | 0 |

Preprocessing differs slightly by model type:

| Model Type | Numeric Processing | Categorical Processing |
|---|---|---|
| Linear and logistic models | Median imputation and standard scaling | Constant imputation with `Missing`, then one-hot encoding |
| Tree-based models | Median imputation without scaling | Constant imputation with `Missing`, then one-hot encoding |

### 3.4 Models Evaluated

| Model | Purpose |
|---|---|
| Zero Revenue Baseline | Predicts zero revenue for every session |
| Mean Log Revenue Baseline | Predicts the training-set mean log revenue |
| Linear Regression | Simple linear supervised baseline |
| Random Forest Regressor | Nonlinear ensemble baseline |
| XGBoost Regressor | Gradient boosting direct regression model |
| LightGBM Regressor | Gradient boosting direct regression model |
| Two-Step Logistic + Random Forest | Classifies purchase first, then predicts positive revenue amount |
| Two-Step Logistic + XGBoost | Two-step model with XGBoost positive-revenue regressor |
| Two-Step Logistic + LightGBM | Two-step model with LightGBM positive-revenue regressor |

### 3.5 Final Modeling Strategy

The final notebook compares direct revenue regression models against two-step models. The two-step models estimate purchase probability with logistic regression, then multiply that probability by a positive-revenue prediction.

Although the target is highly zero-inflated, the direct gradient boosting models performed better than the two-step models. The final recommended model is **LightGBM Regressor** because it achieved the lowest test RMSE.

---

## 4. Results
[Back to Top](#top)

### 4.1 Final Model Performance

All results are measured on the held-out test set using log-transformed revenue. Lower RMSE is better.

| Rank | Model | RMSE on Log Revenue | MAE on Log Revenue | Notes |
|---:|---|---:|---:|---|
| 1 | **LightGBM Regressor** | **1.5192** | 0.2667 | Best overall model |
| 2 | **XGBoost Regressor** | **1.5210** | 0.2673 | Nearly tied with LightGBM |
| 3 | Random Forest Regressor | 1.5241 | 0.2635 | Strong nonlinear baseline |
| 4 | Linear Regression | 1.7320 | 0.4140 | Better than naive RMSE baselines, but weaker than tree models |
| 5 | Mean Log Revenue Baseline | 1.8284 | 0.3788 | Predicts the same mean log revenue for every session |
| 6 | Zero Revenue Baseline | 1.8382 | **0.1896** | Lowest MAE, but poor RMSE because it misses revenue sessions |
| 7 | Two-Step Logistic + Random Forest | 3.7540 | 1.5121 | Underperformed due to unstable purchase probabilities |
| 8 | Two-Step Logistic + XGBoost | 3.7567 | 1.5156 | Two-step model did not improve performance |
| 9 | Two-Step Logistic + LightGBM | 3.7580 | 1.5168 | Two-step model did not improve performance |

### 4.2 Main Findings

**LightGBM achieved the strongest overall performance.**  
LightGBM achieved the best RMSE (1.5192), narrowly ahead of XGBoost (1.5210). This confirms that gradient boosting models are well suited for the nonlinear patterns in this tabular revenue prediction problem.

**XGBoost performed nearly identically to LightGBM.**  
XGBoost achieved an RMSE of 1.5210, which was only 0.0018 higher than LightGBM. Both models outperformed Random Forest on RMSE.

**Random Forest remained a strong baseline.**  
Random Forest achieved an RMSE of 1.5241. This shows that tree-based models can capture useful nonlinear relationships among engagement, traffic source, device, geography, timing, and visitor history features.

**MAE is misleading for this dataset.**  
The zero-revenue baseline had the lowest MAE because most sessions have no revenue. However, it performed worse on RMSE because it completely misses the rare high-revenue sessions that matter most.

**Two-step models did not improve results.**  
The two-step models performed much worse than the direct regressors. The likely reason is that the logistic regression purchase probabilities were poorly calibrated on a target where only about 1.08% of sessions generated revenue. Those unstable probabilities were multiplied into the revenue predictions and increased error.

**Leakage-safe visitor history is important.**  
The final notebook includes prior-session visitor features because EDA showed that previous visitor behavior is related to future revenue. These features were built using only earlier sessions, which protects the evaluation from future-information leakage.

---

## 5. Discussion
[Back to Top](#top)

### 5.1 Why RMSE on Log Revenue?

RMSE on `log1p(transactionRevenue)` is the primary metric because it penalizes larger mistakes more strongly than MAE while reducing the extreme scale of raw revenue. This matters because a useful business model should identify rare revenue-generating sessions, not simply predict zero for nearly everyone.

### 5.2 Model Interpretation

The results suggest that revenue prediction depends on feature interactions. For example, the value of a session can depend on a combination of visit number, pageviews, traffic source, device type, country, time, and prior visitor behavior. Gradient boosting models are effective here because they can capture nonlinear relationships without manually creating every interaction.

### 5.3 Current Limitations

| Limitation | Impact |
|---|---|
| Revenue-generating sessions are rare | Purchase patterns are difficult to learn reliably |
| Positive revenue is highly skewed | Large purchases remain difficult to predict accurately |
| Results use one visitor-grouped train/test split | Additional validation could improve confidence in the final ranking |
| High-cardinality categories are grouped | Some category-level detail is lost when rare values become `Other` |
| Visitor history features depend on prior sessions | New visitors have limited historical signal |
| Two-step models used logistic regression for purchase probability | Better calibration may be needed before a two-step design can help |

### 5.4 Business Relevance

A useful session revenue model can help an e-commerce team prioritize marketing resources, evaluate traffic quality, improve retargeting, and identify sessions or visitors that are more likely to generate meaningful revenue. The main value is not predicting every zero-revenue session correctly. The main value is identifying the small share of sessions that are most likely to produce revenue.

---

## 6. Project Completion Summary
[Back to Top](#top)

The final project successfully completed the full machine learning workflow:

- Loaded and processed the full `train_v2.csv` dataset with Polars.
- Extracted useful fields from nested JSON-like columns.
- Performed EDA on target skew, zero inflation, missingness, outliers, categorical patterns, time effects, and visitor-level behavior.
- Created a clean modeling table with engagement, traffic, device, geography, timing, and visitor history features.
- Built leakage-safe prior-session visitor features.
- Used visitor-grouped train/test splitting to prevent visitor overlap.
- Compared naive baselines, Linear Regression, Random Forest, XGBoost, LightGBM, and two-step models.
- Identified LightGBM as the best final model with a test RMSE of 1.5192.

The final results show that direct gradient boosting regression, especially LightGBM and XGBoost, is the strongest approach for this session-level revenue prediction task.

---

## 7. Statement of Collaboration
[Back to Top](#top)

**Pooja Panchal**  
Led project organization and contributed to data engineering, JSON extraction, report writing, README development, and final notebook support.

**Jinxin Xiao**  
Contributed to EDA analysis and implemented the LightGBM modeling work.

**Justin Chanthabandith**  
Contributed to Kaggle data access, EDA structure, chart interpretation, modeling pipeline design, and XGBoost implementation.

All team members contributed to the final notebook narrative, model comparison, and result interpretation.

---

## Repository Structure
[Back to Top](#top)

```bash
.
|-- data
|   `-- train_v2.csv
|-- notebooks
|   |-- Milestone_3.ipynb
|   `-- Milestone_4.ipynb
|-- reports
|   `-- figures
|-- README.md
`-- requirements.txt
```

---

## How to Run
[Back to Top](#top)

### 1. Install dependencies

```bash
pip install kaggle pandas polars pyarrow scikit-learn matplotlib xgboost lightgbm jupyter
```

### 2. Download the data

Use one of the following options:

- Run the Kaggle download cells inside `Milestone_4.ipynb` after adding your Kaggle API token.
- Download `train_v2.csv` manually from Kaggle and place it where the notebook expects it.

The final notebook uses:

```python
DATA_PATH = "train_v2.csv"
```

If your data is stored in another folder, update `DATA_PATH` before running the data loading cell.

### 3. Open the final notebook

```bash
jupyter notebook notebooks/Milestone_4.ipynb
```

### 4. Run the notebook cells in order

The final notebook includes:

1. Kaggle setup and data loading
2. JSON field extraction
3. Target engineering
4. Exploratory data analysis
5. Leakage-safe visitor feature engineering
6. Visitor-grouped train/test split
7. Model training and evaluation
8. Final results and interpretation

---

## References
[Back to Top](#top)

1. Kaggle. *Google Analytics Customer Revenue Prediction*. https://www.kaggle.com/c/ga-customer-revenue-prediction
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Proceedings of NeurIPS.
5. Scikit-learn Developers. *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org
6. Polars Developers. *Polars Documentation*. https://pola.rs
