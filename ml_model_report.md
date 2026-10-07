# Machine Learning Model Documentation & Performance Report

## 1. Executive Summary
This document reports the machine learning system engineered for the **College Placement Readiness Predictor**. The AI system evaluates candidates across academic credentials, coding practice, open-source contribution metrics, professional social presence, and project complexity to deliver:
1. **Placement Readiness Score** ($0\% - 100\%$)
2. **Target Placement Tier**:
   - **Tier 0**: Not Ready (< 52% score)
   - **Tier 1**: Service Tier Ready (52% - 68% score)
   - **Tier 2**: Product Tier Ready (68% - 83% score)
   - **Tier 3**: Dream Tier / Tier-1 Ready (≥ 83% score)

---

## 2. Dataset Information
- **Total Benchmark Profiles**: 2500 student records.
- **Dataset Files**: `placement_dataset.csv` and `placement_dataset_info.xlsx`.
- **Evaluated Input Features**:
  1. `cgpa`: Academic Grade Point Average (5.0 - 10.0 scale)
  2. `leetcode_rating` & `leetcode_solved`: Data Structures & Algorithms proficiency
  3. `github_repos`, `github_commits_year`, `github_stars`, `github_score`: Open-source activity & Git profile strength
  4. `linkedin_connections`, `linkedin_certs`, `linkedin_posts_freq`, `linkedin_score`: Professional networking metric
  5. `projects_count` & `project_complexity`: Practical software application experience
  6. `skills_count` & `skill_score`: Depth of core technology stack capabilities
  7. `dept`: Engineering branch / department

---

## 3. Machine Learning Algorithms Evaluated & Purpose

### A. Polynomial Ridge Regressor + Tier Pipeline
- **Algorithm Type**: Regularized $L_2$ Ridge Regression with Polynomial Feature Expansion.
- **Purpose**: Fits linear and interaction weights across scaled candidate metrics while penalizing extreme coefficients.
- **Why Selected**: Outperformed all models by achieving **`94.20%` accuracy** and high $R^2$ score alignment.

### B. Multi-Class Softmax Classifier
- **Algorithm Type**: Softmax / One-vs-Rest Multinomial Logistic Regression.
- **Purpose**: Directly models probability distributions across placement tier categories using cross-entropy loss.
- **Why Evaluated**: Serves as direct probabilistic classifier baseline.

### C. Gradient Boosting Ensemble
- **Algorithm Type**: Sequential Boosting Machine.
- **Purpose**: Iteratively fits residual error vectors across features to minimize mean squared prediction error.
- **Why Evaluated**: Evaluates non-linear boosting performance.

### D. K-Nearest Neighbors (KNN)
- **Algorithm Type**: Non-parametric Instance-Based Classifier.
- **Purpose**: Measures Euclidean distance to $k=7$ nearest student profiles in normalized feature space.
- **Why Evaluated**: Provides benchmark instance similarity matching.

---

## 4. Model Performance Comparison

### Classification Performance (Placement Tier Prediction)

| Machine Learning Algorithm | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Polynomial Ridge Regressor + Tier Pipeline** | **94.20%** | 0.9292 | 0.9205 | 0.9246 | **SELECTED BEST** |
| **Multi-Class Softmax Classifier** | **92.00%** | 0.9433 | 0.7173 | 0.7446 | Evaluated |
| **Gradient Boosting Ensemble** | **93.00%** | 0.9544 | 0.8461 | 0.8886 | Evaluated |
| **K-Nearest Neighbors (KNN)** | **81.20%** | 0.8761 | 0.6186 | 0.6572 | Evaluated |

### Regression Performance (Readiness Score Percentage Prediction)

| Machine Learning Algorithm | $R^2$ Score | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Polynomial Ridge Regressor** | **0.9853** | 1.21% | 0.98% | **SELECTED BEST** |
| **Gradient Boosting Regressor** | **0.9843** | 1.25% | 1.01% | Evaluated |

---

## 5. Deployed Model & Accuracy
- **Deployed Model**: `Polynomial Ridge Regressor + Tier Pipeline`
- **Classification Accuracy**: **`94.20%`**
- **Regression Model**: `Polynomial Ridge Regressor`

---

## 6. Feature Importance Breakdown

| Ranking | Feature Attribute | Importance Weight | Predictive Contribution |
| :---: | :--- | :---: | :--- |
| #1 | `coding_score` | `0.4212` | High impact on technical and overall placement tier |
| #2 | `project_score` | `0.1534` | High impact on technical and overall placement tier |
| #3 | `dept_boost` | `0.1074` | High impact on technical and overall placement tier |
| #4 | `cgpa` | `0.0551` | High impact on technical and overall placement tier |
| #5 | `cgpa_score` | `0.0551` | High impact on technical and overall placement tier |
| #6 | `skill_score` | `0.0443` | High impact on technical and overall placement tier |
| #7 | `github_score` | `0.0335` | High impact on technical and overall placement tier |
| #8 | `linkedin_connections` | `0.0267` | High impact on technical and overall placement tier |
| #9 | `github_commits_year` | `0.0211` | High impact on technical and overall placement tier |
| #10 | `linkedin_certs` | `0.0204` | High impact on technical and overall placement tier |
| #11 | `linkedin_score` | `0.0161` | High impact on technical and overall placement tier |
| #12 | `linkedin_posts_freq` | `0.0128` | High impact on technical and overall placement tier |
| #13 | `github_repos` | `0.0078` | High impact on technical and overall placement tier |
| #14 | `skills_count` | `0.0075` | High impact on technical and overall placement tier |
| #15 | `github_stars` | `0.0074` | High impact on technical and overall placement tier |
| #16 | `projects_count` | `0.0034` | High impact on technical and overall placement tier |
| #17 | `leetcode_solved` | `0.0031` | High impact on technical and overall placement tier |
| #18 | `project_complexity` | `0.0027` | High impact on technical and overall placement tier |
| #19 | `leetcode_rating` | `0.0009` | High impact on technical and overall placement tier |

---
*College Placement Readiness Predictor — Automated ML Report.*
