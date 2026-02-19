# Distributed Movie Recommendation System (PySpark + ALS)

A distributed movie recommendation engine built using **PySpark and Spark MLlib (ALS)** in a Databricks environment.  
This project demonstrates scalable data processing, collaborative filtering, cross-validation tuning, and evaluation using both regression and ranking metrics.

---

## 1. Project Overview

This project implements an end-to-end recommendation workflow including:

- Distributed data processing with Spark
- Exploratory Data Analysis (EDA)
- Collaborative Filtering using Alternating Least Squares (ALS)
- Train/Test split comparison
- Cross-validation and hyperparameter tuning
- Model evaluation using RMSE, MAE, Precision, Recall, and F1
- Personalized recommendations for selected users

The objective is to evaluate how data splits and hyperparameters affect recommendation performance in a distributed environment.

---

## 2. System Architecture

Data → Spark DataFrame → ALS Model → Cross-Validation → Evaluation → Recommendations

All computations were executed in a distributed Spark environment (Databricks).

---

## 3. Dataset

Movie ratings dataset (course-provided).

- UserID
- MovieID
- Rating
- Timestamp

⚠️ Dataset is not included in this repository.

---

## 4. Exploratory Data Analysis

Key insights:
- Distribution of ratings across users and movies
- Identification of highly active users
- Rating sparsity analysis
- Top movies by average rating

EDA was performed using Spark DataFrames with visualization support.

---

## 5. Model Implementation

### 5.1 Collaborative Filtering

Algorithm: **ALS (Alternating Least Squares)**  
Library: `pyspark.ml.recommendation.ALS`

Key characteristics:
- Handles large-scale sparse matrices
- Parallelized across Spark cluster
- Implicit feedback handling available (not used in this version)

---

### 5.2 Train/Test Split Comparison

The dataset was evaluated under multiple split ratios:

- 60/40
- 70/30
- 75/25
- 80/20

Performance was compared using RMSE to determine optimal configuration.

Best observed configuration:
- 80/20 split
- RMSE ≈ 0.93

---

### 5.3 Hyperparameter Tuning

Cross-validation performed over:

- rank ∈ {8, 10, 12}
- regParam ∈ {0.05, 0.1, 0.15}
- maxIter ∈ {10, 15}

Grid search was used to evaluate parameter impact on RMSE.

Insights:
- Increasing rank improved representational capacity but increased training time
- Higher regularization reduced overfitting but may increase bias
- Trade-off between model complexity and runtime was observed

---

## 6. Model Evaluation

### Regression Metrics
- RMSE
- MAE
- MSE

### Ranking Metrics (Threshold-based relevance)
- Precision
- Recall
- F1 Score

Example observation:
- Precision ≈ 0.75
- Recall ≈ 0.14
- F1 ≈ 0.24

High precision but lower recall indicates the model favors confident recommendations.

---

## 7. Personalized Recommendations

Recommendations were generated for selected user IDs using:

```python
model.recommendForAllUsers()
