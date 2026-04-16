# DSC288-Capstone-Project
<a id="top"></a>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E3A8A,100:0F766E&height=180&section=header&text=Predicting%20Customer%20Revenue&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Web%20Analytics%20Data%20%7C%20Single-Stage%20Regression%20vs%20Two-Step%20Models&descAlignY=55&descSize=18" />
</p>

<div align="center">
  <h3><i>A Comparison of Single-Stage Regression and Two-Step Machine Learning Models</i></h3>
  <h4>DSC 288R: Capstone Project</h4>

  <p>
    <strong>Pooja Panchal</strong> &nbsp;&bull;&nbsp;
    <strong>Jinxin Xiao</strong> &nbsp;&bull;&nbsp;
    <strong>Justin Chanthabandith</strong>
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

This project is well suited for DSC 288R because the dataset contains approximately 1.7 million session-level records with structured and semi-structured fields. Preparing the data requires flattening nested Google Analytics columns, handling missing values, engineering temporal and behavioral features, and encoding categorical variables at scale. :contentReference[oaicite:0]{index=0}

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

The project uses the **Google Analytics Customer Revenue Prediction** dataset from Kaggle. The primary training file, `train_v2.csv`, contains roughly **1.7 million session-level records** with nested fields related to traffic source, device information, geographic attributes, visit timing, and user behavior. :contentReference[oaicite:1]{index=1}

| Dataset | Measures | Resolution |
|---|---|---|
| `train_v2.csv` | Session metadata, traffic source, device, geography, activity, transaction revenue | Session-level |

Key EDA goals include:
- quantifying the proportion of zero-revenue versus positive-revenue sessions
- understanding missingness across nested and categorical fields
- profiling major traffic sources, device categories, and geographic patterns
- examining skewness in the revenue distribution
- identifying temporal trends in visits and purchases

Key EDA findings from distributed Spark operations such as `df.count()`, `df.describe()`, `groupBy().agg()`, and `distinct().count()` will include:

- the dataset is heavily imbalanced, with many zero-revenue sessions
- transaction revenue is highly right-skewed
- several important predictors are nested and must be flattened before modeling
- user behavior, traffic source, device type, and visit timing appear likely to contribute predictive signal

**Figures**
- Revenue distribution before and after log transformation
- Class balance for zero vs positive revenue
- Top traffic sources and device categories
- Missing value summary across selected features

---

### 2.2 Preprocessing

Before modeling, the raw Google Analytics data will be transformed into an analysis-ready table through a preprocessing pipeline that includes:

- flattening nested JSON-like columns
- handling missing values
- encoding categorical features
- creating date-based and session-level features
- selecting relevant predictors for modeling

Additional feature engineering will focus on:
- temporal variables derived from visit dates
- traffic source information
- device and browser categories
- geographic attributes
- behavioral session indicators

Because the revenue target is sparse and highly skewed, the preprocessing workflow will also prepare the data for two separate modeling strategies:
1. direct regression on revenue
2. a two-step pipeline with purchase classification followed by regression on purchasing sessions only

### 2.3 Models

#### Baseline
**Random Forest**
- Serves as an interpretable and robust ensemble baseline for tabular prediction
- Useful for comparing against boosting-based methods

#### Model 1
**Single-Stage Regression**
- A direct model trained to predict customer revenue from session-level features
- Designed to test whether one unified regression approach can handle sparse revenue targets

#### Model 2
**Two-Step Classification + Regression Pipeline**
- Step 1: classifier predicts whether a session will generate any revenue
- Step 2: regressor estimates revenue only for predicted purchasing sessions

This architecture is motivated by the large number of zero-revenue sessions in the data and is intended to better handle zero inflation. :contentReference[oaicite:2]{index=2}

#### Primary Algorithms
**XGBoost**
- Well suited for structured and sparse tabular data
- Handles nonlinear feature interactions effectively

**LightGBM**
- Efficient gradient boosting model with strong performance on large tabular datasets
- Particularly useful for high-dimensional feature spaces and scalable training

### 2.4 Tools and Technical Stack

| Category | Tools |
|---|---|
| **Programming Language** | Python |
| **Distributed Processing** | Apache Spark |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Data Source** | Kaggle |
| **Data Format Work** | Nested field flattening, tabular feature engineering |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebook Development** | Jupyter |
| **Version Control** | Git, GitHub |

---

## 3. Results
[Back to Top](#top)

### 3.1 Model Performance

| Model | RMSE on Log Revenue | Notes |
|---|---:|---|
| Random Forest | TBD | Baseline ensemble model |
| Single-Stage Regression | TBD | Direct revenue prediction |
| Two-Step Model | TBD | Classification followed by regression |
| XGBoost | TBD | Boosted tree model |
| LightGBM | TBD | Efficient gradient boosting model |

### 3.2 Visual Results
- Predicted vs actual revenue
- Residual plots
- Feature importance rankings
- Comparison of single-stage and two-step performance
- Purchase classification performance for the first stage of the two-step model

### 3.3 Best Model
This section will summarize which modeling strategy performs best on held-out validation data and whether explicitly separating purchase prediction from revenue estimation improves predictive performance.

---

## 4. Discussion
[Back to Top](#top)

### 4.1 Metric Selection

The primary evaluation metric is **RMSE on log-transformed revenue**. This is appropriate because the target variable is continuous, strongly skewed, and affected by large outliers. Log transformation reduces the influence of extreme purchase values and provides a more stable basis for model comparison. :contentReference[oaicite:3]{index=3}

### 4.2 Model Interpretation

Model interpretation will focus on feature importance analysis to identify which session-level characteristics are most associated with purchasing behavior and revenue generation. Likely important predictors include traffic source, device type, timing variables, and session activity features.

### 4.3 Shortcomings

Several limitations are expected:
- severe class imbalance due to many zero-revenue sessions
- possible information loss from session-level aggregation
- missing or noisy values in nested analytics fields
- sensitivity of results to feature engineering and target transformation choices

### 4.4 Business Relevance

Even modest gains in predictive accuracy can support better customer targeting, campaign optimization, and strategic planning in e-commerce environments. This makes the problem practically important in addition to being methodologically interesting.

### 4.5 Impact of Distributed Computing

Distributed computing is especially useful in this project because it supports scalable parsing, flattening, cleaning, grouping, and feature engineering across a large session-level dataset. Spark can also accelerate EDA and repeated experiments across multiple model pipelines.

---

## 5. Conclusion
[Back to Top](#top)

### What We Learned
- Whether web analytics features contain sufficient signal to predict revenue
- Whether a two-step architecture better handles zero-inflated revenue outcomes
- Which session-level predictors contribute most to purchasing behavior

### What We Would Do Differently
- add more extensive hyperparameter tuning
- explore additional feature selection strategies
- evaluate alternative target transformations
- incorporate richer user-level session history if available

### What We Would Explore With More Time
- CatBoost or neural tabular models
- calibrated probability estimates for the first-stage classifier
- ensembling between single-stage and two-step approaches
- more detailed temporal modeling of user sessions

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
