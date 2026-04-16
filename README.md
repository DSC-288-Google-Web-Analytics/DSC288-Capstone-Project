<a id="top"></a>

<div align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:A7C7E7,25:F4B6C2,50:F9E79F,75:B7E4C7,100:CDB4DB&height=190&section=header&text=Predicting%20Customer%20Revenue&fontSize=32&fontColor=2F3E46&animation=fadeIn&fontAlignY=33&desc=Google%20Web%20Analytics%20%7C%20Single-Stage%20Regression%20vs%20Two-Step%20Models&descAlignY=54&descSize=15"
    style="display: block; margin: 0 auto;"
  />
</div>
<div align="center">
  <h3><i>A Comparison of Single-Stage Regression and Two-Step Machine Learning Models</i></h3>
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
    <img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logoColor=white" />
    <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  </div>
</div>

---

<p align="center">
  <a href="#1-introduction" style="font-size: 16px;">Introduction</a> |
  <a href="#2-methods" style="font-size: 16px;">Methods</a> |
  <a href="#3-results" style="font-size: 16px;">Results</a> |
  <a href="#4-discussion" style="font-size: 16px;">Discussion</a> |
  <a href="#5-conclusion" style="font-size: 16px;">Conclusion</a> |
  <a href="#6-statement-of-collaboration" style="font-size: 16px;">Collaboration</a>
</p>

---

## 1. Introduction
[Back to Top](#top)

### Why This Project?

E-commerce companies collect large volumes of web analytics data describing how users arrive at a site, what devices they use, how they navigate sessions, and whether those visits lead to purchases. Translating this behavioral data into revenue forecasts is an important machine learning problem because accurate predictions can improve customer targeting, advertising strategy, and overall business planning.

This project uses the Google Analytics Customer Revenue Prediction dataset to study whether session-level web analytics features can be used to predict customer revenue. Because most sessions generate zero revenue while a smaller fraction lead to purchases, this problem is both practically relevant and technically challenging. 

### Why Big Data and Distributed Computing?

This project is well suited for DSC 288R because the dataset contains approximately 1.7 million session-level records with structured and semi-structured fields. Preparing the data requires flattening nested Google Analytics columns, handling missing values, engineering temporal and behavioral features, and encoding categorical variables at scale. 

Distributed computing tools such as Spark can support large-scale preprocessing, feature extraction, aggregation, and exploratory analysis more efficiently than a single-machine workflow. The project also involves repeated experimentation across multiple machine learning pipelines, making scalable data processing and reproducible workflow design especially valuable. 

### Project Overview

| Aspect | Detail |
|---|---|
| **Problem Type** | Regression with zero-inflated target structure |
| **Target Variable** | Customer transaction revenue |
| **Dataset** | Google Analytics Customer Revenue Prediction (`train_v2.csv`) |
| **Research Question** | Does a two-step model outperform a single-stage regression model for customer revenue prediction? |
| **Baseline Model** | Random Forest |
| **Model 1** | Single-stage revenue regression |
| **Model 2** | Two-step classification + regression pipeline |
| **Advanced Models** | XGBoost, LightGBM |
| **Evaluation** | RMSE on log-transformed revenue |
| **Secondary Analysis** | Feature importance and interpretation |
| **Infrastructure** | Python, Spark, Kaggle dataset, distributed preprocessing workflow |

---

## 2. Methods
[Back to Top](#top)

### 2.1 Data Exploration

The project uses the **Google Analytics Customer Revenue Prediction** dataset from Kaggle. The primary training file, `train_v2.csv`, contains roughly **1.7 million session-level records** with nested fields related to traffic source, device information, geographic attributes, visit timing, and user behavior. :contentReference[oaicite:4]{index=4}

| Dataset | Measures | Resolution |
|---|---|---|
| `train_v2.csv` | Session metadata, traffic source, device, geography, activity, transaction revenue | Session-level |

At this stage, exploratory data analysis has focused on understanding the structure of the dataset, the distribution of the target variable, and the major feature groups available for modeling. Initial inspection suggests that the revenue outcome is sparse, with many sessions generating no revenue and a much smaller set of positive-revenue sessions. This supports the motivation for comparing direct regression against a two-step modeling strategy. :contentReference[oaicite:5]{index=5}

Current EDA goals and observations include:

- identifying the proportion of zero-revenue versus positive-revenue sessions
- examining the highly right-skewed distribution of transaction revenue
- reviewing missingness patterns across nested and categorical variables
- profiling traffic source, device type, and geographic fields
- identifying which nested columns will need to be flattened before modeling

Key EDA findings from distributed Spark operations such as `df.count()`, `df.describe()`, `groupBy().agg()`, and `distinct().count()`:

- the dataset is heavily imbalanced, with many zero-revenue sessions
- transaction revenue is highly skewed and likely benefits from log transformation
- several important predictors are nested and require preprocessing before modeling
- traffic, behavioral, temporal, and device-related fields appear to provide meaningful predictive signal

**Figures**
- [Insert revenue distribution plot]
- [Insert zero vs positive revenue bar chart]
- [Insert missing value summary]
- [Insert top traffic sources or device categories]

---

### 2.2 Preprocessing

[Placeholder: This section will describe the preprocessing pipeline once finalized.]

Planned topics to include:
- flattening nested JSON-like columns
- handling missing values
- encoding categorical variables
- creating date-based and session-level features
- selecting relevant predictors for modeling

---

### 2.3 Models

[Placeholder: This section will describe the completed modeling workflow.]

Planned model comparisons:
- **Baseline:** Random Forest
- **Model 1:** Single-stage regression
- **Model 2:** Two-step classification + regression
- **Advanced models:** XGBoost and LightGBM

Planned details to include:
- model justification
- feature inputs
- hyperparameter tuning strategy
- train, validation, and test split design

---

### 2.4 Tools and Technical Stack

| Category | Tools |
|---|---|
| **Programming Language** | Python |
| **Distributed Processing** | Apache Spark |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Data Source** | Kaggle |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebook Development** | Jupyter |
| **Version Control** | Git, GitHub |

---

## 3. Results
[Back to Top](#top)

[Placeholder: Results will be added after model training and evaluation are completed.]

### 3.1 Model Performance

| Model | RMSE on Log Revenue | Notes |
|---|---:|---|
| Random Forest | TBD | Baseline ensemble model |
| Single-Stage Regression | TBD | Direct revenue prediction |
| Two-Step Model | TBD | Classification followed by regression |
| XGBoost | TBD | Boosted tree model |
| LightGBM | TBD | Efficient gradient boosting model |

### 3.2 Visual Results
- [Insert predicted vs actual revenue plot]
- [Insert residual plots]
- [Insert feature importance rankings]
- [Insert single-stage vs two-step comparison]

### 3.3 Best Model
[Placeholder: Summarize best-performing model here.]

---

## 4. Discussion
[Back to Top](#top)

[Placeholder: This section will be completed after results are available.]

### 4.1 Metric Selection
[Placeholder: Explain why RMSE on log-transformed revenue is the primary metric.]

### 4.2 Model Interpretation
[Placeholder: Discuss feature importance and model behavior.]

### 4.3 Shortcomings
[Placeholder: Discuss class imbalance, sparse purchases, missing data, and other limitations.]

### 4.4 Business Relevance
[Placeholder: Explain how the results support e-commerce decision-making.]

### 4.5 Impact of Distributed Computing
[Placeholder: Describe how Spark and distributed processing improved scalability.]

---

## 5. Conclusion
[Back to Top](#top)

[Placeholder: This section will be completed after final analysis.]

### What We Learned
- [Placeholder]
- [Placeholder]
- [Placeholder]

### What We Would Do Differently
- [Placeholder]
- [Placeholder]
- [Placeholder]

### What We Would Explore With More Time
- [Placeholder]
- [Placeholder]
- [Placeholder]

---

## 6. Statement of Collaboration
[Back to Top](#top)

**Pooja Panchal (Project Manager, Front End Developer, and Data Engineer)**  
Led project management responsibilities including scheduling, coordination, and overall progress tracking. Contributed to frontend development and README design, and supported data engineering tasks such as preprocessing, feature engineering, and implementation of the end-to-end modeling pipeline.

**Jinxin Xiao (Data Engineer)**  
Contributed to data engineering responsibilities including data preprocessing, feature extraction, and preparation of the final modeling dataset. Supported the machine learning workflow by helping structure data pipelines for training and evaluation.

**Justin Chanthabandith (EDA and Data Engineer)**  
Led exploratory data analysis to better understand data structure, missingness, class imbalance, and feature trends. Also contributed to data engineering tasks including data cleaning, transformation, and feature preparation for downstream modeling.

---

## Repository Structure

```bash
.
├── config
│   ├── main.yaml
│   ├── model
│   │   ├── random_forest.yaml
│   │   ├── xgboost.yaml
│   │   └── lightgbm.yaml
│   └── process
│       ├── flatten.yaml
│       └── preprocess.yaml
├── data
│   ├── raw
│   ├── processed
│   └── final
├── docs
├── models
├── notebooks
├── reports
│   └── figures
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── flatten_ga.py
│   │   └── preprocess.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   ├── train_classifier.py
│   │   ├── train_regressor.py
│   │   ├── train_single_stage.py
│   │   └── evaluate.py
│   └── visualization
│       └── plots.py
├── tests
├── README.md
└── requirements.txt
