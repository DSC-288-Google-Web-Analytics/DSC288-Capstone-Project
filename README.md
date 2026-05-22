<a id="top"></a>

<div align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:A7C7E7,25:F4B6C2,50:F9E79F,75:B7E4C7,100:CDB4DB&height=190&section=header&text=Google%20Analytics%20Revenue%20Prediction&fontSize=32&fontColor=2F3E46&animation=fadeIn&fontAlignY=33&desc=Session-Level%20Revenue%20Forecasting%20%7C%20Zero-Inflated%20Regression%20%7C%20Tree-Based%20ML&descAlignY=54&descSize=15"
    style="display: block; margin: 0 auto;"
  />
</div>

<div align="center">
  <h3><i>Predicting Customer Revenue from Google Analytics Session Data</i></h3>
  <h4>DSC 288R: Capstone Project</h4>

  <p>
    <strong>Pooja Panchal</strong> (Project Manager, Front End Developer, Data Engineer)
    &nbsp;&bull;&nbsp;
    <strong>Jinxin Xiao</strong> (Data Engineer)
    &nbsp;&bull;&nbsp;
    <strong>Justin Chanthabandith</strong> (EDA, Data Engineer)
  </p>

  <div>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white" />
    <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  </div>
</div>

---

<p align="center">
  <a href="#1-introduction">Introduction</a> |
  <a href="#2-data-and-problem-structure">Data</a> |
  <a href="#3-methods">Methods</a> |
  <a href="#4-results-so-far">Results</a> |
  <a href="#5-discussion">Discussion</a> |
  <a href="#6-next-steps">Next Steps</a> |
  <a href="#7-statement-of-collaboration">Collaboration</a>
</p>

---

## 1. Introduction
[Back to Top](#top)

### Why This Project?

E-commerce companies collect large volumes of web analytics data describing how visitors arrive at a site, what devices they use, how they navigate each session, and whether those sessions lead to purchases. This project uses the **Google Analytics Customer Revenue Prediction** dataset to estimate how much revenue a single website session will generate.

The task is difficult because most sessions produce no revenue. This creates a **zero-inflated regression problem**: the model must learn both whether a purchase is likely and, if a purchase happens, how large the revenue might be.

### Research Question

> Can session-level Google Analytics features predict customer revenue, and do two-step purchase-and-revenue models outperform direct regression models on a highly zero-inflated target?

### Project Overview

| Aspect | Detail |
|---|---|
| **Problem Type** | Zero-inflated revenue prediction |
| **Target Variable** | `totals.transactionRevenue` |
| **Primary Metric** | RMSE on `log1p(transactionRevenue)` |
| **Dataset** | Google Analytics Customer Revenue Prediction, Kaggle 2018 |
| **Data Scale** | About 1.7 million sessions and 700,000+ visitors |
| **Positive Revenue Rate** | About 1.3% of sessions |
| **Main Challenge** | Most sessions have zero revenue, while positive revenue is highly skewed |
| **Best Milestone 3 Model** | Random Forest Regressor |
| **Current Final Pipeline Direction** | LightGBM two-stage model with purchase probability and revenue regression |

---

## 2. Data and Problem Structure
[Back to Top](#top)

### Dataset

The project uses `train_v2.csv` from Kaggle's **Google Analytics Customer Revenue Prediction** competition. Each row represents one anonymized Google Merchandise Store session.

| Feature Group | Examples |
|---|---|
| Visitor identity | `fullVisitorId` |
| Session timing | `date`, `visitStartTime`, `visitNumber` |
| Traffic | `channelGrouping`, `trafficSource` fields |
| Device | `device.isMobile`, `device.deviceCategory` |
| Geography | `geoNetwork.subContinent`, `geoNetwork.country` |
| Session activity | `totals.hits`, `totals.pageviews`, `totals.transactions` |
| Target | `totals.transactionRevenue` |

The raw file is not flat. Several columns, including `device`, `geoNetwork`, `totals`, and `trafficSource`, are stored as nested JSON-like strings, so preprocessing is required before modeling.

### Key Dataset Challenges

| Challenge | Why It Matters | How We Address It |
|---|---|---|
| **Zero inflation** | About 98.7% of sessions have no revenue | Compare naive baselines, direct regression, and two-step modeling |
| **Revenue skew** | A few purchases are much larger than most | Use `log1p(transactionRevenue)` for model targets and evaluation |
| **Nested JSON fields** | Important predictors are not directly model-ready | Extract selected fields from `device`, `geoNetwork`, and `totals` |
| **High-cardinality categories** | Country, source, and device fields can create sparse features | Reduce or encode categorical variables carefully |
| **Visitor leakage risk** | The same visitor can appear in multiple sessions | Split by `fullVisitorId` rather than random rows |
| **Metric sensitivity** | MAE rewards predicting zero too often | Use RMSE on log revenue as the primary metric |

---

## 3. Methods
[Back to Top](#top)

### 3.1 Preprocessing Pipeline

The current workflow prepares the raw session data through the following steps:

1. **Efficient loading**  
   `final.py` uses Polars lazy CSV scanning to load selected columns from `train_v2.csv`.

2. **JSON parsing and flattening**  
   The `safe_json()` and `column_split()` functions extract model-ready fields from nested columns:
   - `device`: `isMobile`, `deviceCategory`
   - `geoNetwork`: `subContinent`, `country`
   - `totals`: `visits`, `hits`, `pageviews`, `transactions`, `transactionRevenue`

3. **Target engineering**  
   Missing revenue values are treated as zero. The pipeline creates:
   - `has_revenue`: binary target for purchase classification
   - `log1p(transactionRevenue)`: regression target

4. **Date and time conversion**  
   `visitStartTime` is converted from Unix time, and `date` is split into `year`, `month`, and `day`.

5. **Visitor-level splitting**  
   The final script splits users into train, validation, and test groups using unique `fullVisitorId` values. This prevents the same visitor from appearing across multiple splits.

### 3.2 Feature Set in `final.py`

The current final script uses a focused feature set:

| Feature Type | Features |
|---|---|
| Channel | `channelGrouping` |
| Session behavior | `visitNumber` |
| Device | `isMobile`, `deviceCategory` |
| Geography | `subContinent`, `country` |
| Time | `year`, `month`, `day` |

Categorical fields are converted to pandas `category` dtype so LightGBM can use them directly.

### 3.3 Models Tested in Milestone 3

| Model | Purpose |
|---|---|
| Zero Revenue Baseline | Predicts zero revenue for every session |
| Mean Log Revenue Baseline | Predicts the training-set mean log revenue |
| Linear Regression | Simple supervised baseline |
| Random Forest Regressor | Captures nonlinear feature interactions |
| Two-Step Logistic Regression + Random Forest | Classifies purchase, then predicts purchase amount |

### 3.4 Current Final Model Implementation

The current `final.py` implementation moves the two-step idea toward a stronger tree-based approach using LightGBM:

| Stage | Model | Goal |
|---|---|---|
| Stage 1 | `LGBMClassifier` | Estimate probability that a session generates revenue |
| Stage 2 | `LGBMRegressor` | Predict log revenue amount |
| Final Prediction | `P(purchase) * predicted_revenue` | Estimate expected revenue |

This is a useful next step because the Milestone 3 two-step model underperformed when the classifier used logistic regression. A gradient boosting classifier should be better suited to nonlinear patterns, class imbalance, and feature interactions.

---

## 4. Results So Far
[Back to Top](#top)

### 4.1 Milestone 3 Model Performance

Lower RMSE is better.

| Rank | Model | RMSE on Log Revenue | MAE on Log Revenue | Interpretation |
|---:|---|---:|---:|---|
| 1 | **Random Forest Regressor** | **1.524** | 0.263 | Best overall model so far |
| 2 | Linear Regression | 1.732 | 0.414 | Beats naive baselines, but misses nonlinear interactions |
| 3 | Mean Log Revenue Baseline | 1.828 | 0.379 | Predicts the same mean value for every session |
| 4 | Zero Revenue Baseline | 1.838 | **0.190** | Low MAE but poor RMSE because it misses high-revenue sessions |
| 5 | Two-Step Logistic + RF | 3.754 | 1.512 | Underperformed due to unstable purchase probabilities |

### 4.2 Main Findings

**Random Forest is the best model so far.**  
The Random Forest Regressor reduced RMSE by about 17% compared with the zero-revenue baseline. This supports the hypothesis that session revenue depends on nonlinear interactions among channel, device, geography, behavior, and visitor history.

**Linear Regression still learned useful signal.**  
Linear Regression beat both naive baselines, which means the feature set contains real predictive information. However, its gap behind Random Forest suggests that the problem is not purely linear.

**MAE is misleading for this dataset.**  
The zero-revenue baseline has the best MAE because most sessions are zero. That does not mean it is the best business model. It fails to identify the rare high-revenue sessions that matter most.

**The original two-step model was weaker than expected.**  
The Milestone 3 two-step model used Logistic Regression to estimate purchase probability, followed by Random Forest for revenue amount. Because only about 1.3% of sessions generate revenue, the classifier likely overestimated purchase probability for many zero-revenue sessions. Multiplying inflated purchase probabilities by predicted revenue created systematic overprediction.

**The final implementation addresses this weakness.**  
The current `final.py` replaces the logistic first stage with `LGBMClassifier` and uses `LGBMRegressor` for the revenue stage. This keeps the two-step structure but uses models that are better matched to nonlinear tabular data.

---

## 5. Discussion
[Back to Top](#top)

### 5.1 Why RMSE on Log Revenue?

RMSE on log-transformed revenue is the primary metric because it penalizes large misses more strongly than MAE while reducing the extreme scale of raw transaction revenue. This matters because business value comes from identifying rare high-revenue sessions, not merely predicting zero for the majority class.

### 5.2 Model Interpretation

The results suggest that revenue prediction depends on feature interactions rather than isolated variables. For example, the effect of `visitNumber` may depend on channel, device type, geography, and prior behavior. Tree-based models are appropriate because they can capture these conditional relationships without manually creating every interaction term.

### 5.3 Current Limitations

| Limitation | Impact |
|---|---|
| Positive sessions are extremely rare | Classification models can become poorly calibrated |
| Current final script uses a smaller feature set than the full Milestone 3 notebook | Some predictive signal may still be unused |
| Two-step predictions are sensitive to the purchase-probability model | Poor calibration can cascade into revenue overprediction |
| User split is currently randomized by unique visitor IDs | Reproducibility can improve by setting a fixed random seed before shuffling |
| Final LightGBM results are not yet documented in the README | The current README should distinguish proven Milestone 3 results from in-progress final implementation |

### 5.4 Business Relevance

A useful model can help an e-commerce business prioritize sessions or users that are more likely to generate revenue. This can support marketing allocation, campaign evaluation, retargeting, and personalization. The main value is not predicting every zero-revenue session correctly. The main value is finding the smaller set of sessions that are likely to produce meaningful revenue.

---

## 6. Next Steps
[Back to Top](#top)

Before final submission, the strongest improvements are:

1. **Run and report final LightGBM results**  
   Add validation and test AUC for the classifier, plus RMSE for the expected revenue prediction.

2. **Calibrate purchase probabilities**  
   Test Platt scaling or isotonic calibration to prevent inflated purchase probabilities.

3. **Tune the decision threshold**  
   Evaluate whether a hard threshold version improves or worsens RMSE compared with expected revenue.

4. **Add feature importance analysis**  
   Report which channel, device, geography, time, and behavior features are most predictive.

5. **Expand features carefully**  
   Consider adding pageviews, hits, transactions, source or medium, and leakage-safe visitor history features.

6. **Improve reproducibility**  
   Set a random seed before visitor splitting and document the train, validation, and test split sizes.

7. **Add visual diagnostics**  
   Include predicted vs. actual plots, residual plots, and precision-recall curves for the classification stage.

---

## 7. Statement of Collaboration
[Back to Top](#top)

**Pooja Panchal (Project Manager, Front End Developer, and Data Engineer)**  
Led project management responsibilities including scheduling, coordination, and overall progress tracking. Contributed to README design, project presentation, preprocessing, feature engineering, and implementation support for the end-to-end modeling pipeline.

**Jinxin Xiao (Data Engineer)**  
Contributed to data engineering responsibilities including preprocessing, feature extraction, and preparation of model-ready datasets. Supported the machine learning workflow by helping structure data pipelines for training and evaluation.

**Justin Chanthabandith (EDA and Data Engineer)**  
Led exploratory data analysis to understand data structure, missingness, class imbalance, target skew, and feature trends. Contributed to data cleaning, transformation, feature preparation, and result interpretation.

---

## Repository Structure

```bash
.
├── data
│   ├── raw
│   ├── processed
│   └── final
├── notebooks
│   └── Milestone_3.ipynb
├── reports
│   ├── figures
│   └── 2nd Progress Report Presentation.pdf
├── src
│   ├── data
│   ├── features
│   ├── models
│   └── visualization
├── final.py
├── README.md
└── requirements.txt
```

---

## How to Run

1. Download the Kaggle dataset and place `train_v2.csv` in `data/`.
2. Install the project dependencies.
3. Run the final pipeline:

```bash
python final.py
```

The script will:

- load selected fields from `train_v2.csv`,
- flatten nested JSON columns,
- create classification and regression targets,
- split users into train, validation, and test sets,
- train the LightGBM two-stage model,
- print validation and test classification AUC,
- print validation and test RMSE.

---

## References

1. Kaggle. *Google Analytics Customer Revenue Prediction Competition*, 2018. https://www.kaggle.com/c/ga-customer-revenue-prediction
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Chen, T., and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
5. Scikit-learn Developers. *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org
6. Polars Developers. *Polars Documentation*. https://pola.rs
