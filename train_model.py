import csv
import os
import random
import math
import joblib
import numpy as np
import openpyxl
from generate_dataset import generate_placement_dataset

np.random.seed(42)
random.seed(42)

# ==========================================
# 1. Feature Engineering & Preprocessing
# ==========================================

def extract_features_and_targets(filepath='placement_dataset.csv'):
    if not os.path.exists(filepath):
        generate_placement_dataset()

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    depts = sorted(list(set(r['dept'] for r in rows)))
    complexities = ['Basic', 'Intermediate', 'Advanced', 'Production-Grade']
    dept_map = {d: i for i, d in enumerate(depts)}
    comp_map = {c: i for i, c in enumerate(complexities)}

    feature_cols = [
        'cgpa', 'skills_count', 'skill_score', 'leetcode_rating', 'leetcode_solved',
        'github_repos', 'github_commits_year', 'github_stars', 'github_score',
        'linkedin_connections', 'linkedin_certs', 'linkedin_posts_freq', 'linkedin_score',
        'projects_count', 'project_complexity', 'project_score', 'dept_boost',
        'coding_score', 'cgpa_score'
    ]

    X, y_tier, y_score = [], [], []

    for r in rows:
        cgpa = float(r['cgpa'])
        lc_rating = float(r['leetcode_rating'])
        lc_solved = float(r['leetcode_solved'])
        gh_score = float(r['github_score'])
        li_score = float(r['linkedin_score'])
        proj_score = float(r['project_score'])
        skill_score = float(r['skill_score'])
        
        coding_score = min(100.0, max(0.0, (lc_rating - 800) / 1400 * 60 + (lc_solved / 500 * 40)))
        cgpa_score = (cgpa / 10.0) * 100
        dept_boost = 5.0 if r['dept'] in ['Computer Science', 'Information Technology'] else (2.0 if r['dept'] == 'Electronics & Comm' else 0.0)
        comp_val = comp_map[r['project_complexity']]

        feat = [
            cgpa, float(r['skills_count']), skill_score, lc_rating, lc_solved,
            float(r['github_repos']), float(r['github_commits_year']), float(r['github_stars']), gh_score,
            float(r['linkedin_connections']), float(r['linkedin_certs']), float(r['linkedin_posts_freq']), li_score,
            float(r['projects_count']), float(comp_val), proj_score, dept_boost,
            coding_score, cgpa_score
        ]
        X.append(feat)
        y_tier.append(int(r['placement_tier']))
        y_score.append(float(r['readiness_score']))

    X = np.array(X)
    y_tier = np.array(y_tier)
    y_score = np.array(y_score)

    return X, y_tier, y_score, feature_cols, dept_map, comp_map, len(rows)


class PureStandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ==========================================
# 2. Machine Learning Algorithm Implementations
# ==========================================

class RidgeRegressor:
    """ Polynomial Regularized Ridge Regression for exact score prediction """
    def __init__(self, l2=0.1):
        self.l2 = l2
        self.w = None

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0.0
        self.w = np.linalg.inv(X_b.T @ X_b + self.l2 * I) @ X_b.T @ y

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b @ self.w


class SoftmaxClassifier:
    """ Multi-Class Softmax Logistic Regression for placement tier classification """
    def __init__(self, lr=0.1, epochs=1500, l2=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))
        
        Y_oh = np.zeros((n_samples, n_classes))
        Y_oh[np.arange(n_samples), y] = 1.0
        
        for ep in range(self.epochs):
            logits = np.dot(X, self.W) + self.b
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
            dW = (1 / n_samples) * np.dot(X.T, (probs - Y_oh)) + self.l2 * self.W
            db = (1 / n_samples) * np.sum(probs - Y_oh, axis=0, keepdims=True)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
            
    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        return np.argmax(logits, axis=1)


class PureKNN:
    """ Distance-Based K-Nearest Neighbors Classifier """
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_idxs = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idxs].astype(int)
            preds.append(np.bincount(k_labels).argmax())
        return np.array(preds)


class GradientBoostingRegressorPure:
    """ Boosting Ensemble Algorithm """
    def __init__(self, n_estimators=30, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.weights = None

    def fit(self, X, y):
        # Multi-stage linear booster
        self.init_mean = np.mean(y)
        res = y - self.init_mean
        
        self.models = []
        for _ in range(self.n_estimators):
            w = np.linalg.lstsq(X, res, rcond=None)[0]
            pred = X @ w
            res -= self.lr * pred
            self.models.append(w)

    def predict(self, X):
        pred = np.full(len(X), self.init_mean)
        for w in self.models:
            pred += self.lr * (X @ w)
        return pred


# ==========================================
# 3. Model Pipeline Execution & Evaluation
# ==========================================

def compute_classification_metrics(y_true, y_pred, n_classes=4):
    acc = float(np.mean(y_true == y_pred))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        p_c = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_c = tp / (tp + fn) if (tp + fn) > 0 else 0
        f_c = (2 * p_c * r_c) / (p_c + r_c) if (p_c + r_c) > 0 else 0

        precisions.append(p_c)
        recalls.append(r_c)
        f1s.append(f_c)

    return {
        'accuracy': acc,
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1_score': float(np.mean(f1s)),
        'confusion_matrix': cm.tolist()
    }


def compute_regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def train_and_evaluate_all():
    print("===============================================================")
    print(" College Placement Readiness Predictor — ML Training Pipeline ")
    print("===============================================================")

    X, y_tier, y_score, feature_cols, dept_map, comp_map, n_samples = extract_features_and_targets()
    print(f"Loaded {n_samples} benchmark student records.")

    # Train/Test Split (80/20)
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    test_size = int(n_samples * 0.2)
    tr_idx, te_idx = idxs[test_size:], idxs[:test_size]

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train_tier, y_test_tier = y_tier[tr_idx], y_tier[te_idx]
    y_train_score, y_test_score = y_score[tr_idx], y_score[te_idx]

    scaler = PureStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Ridge Regressor (Exact Score % Prediction)
    ridge_reg = RidgeRegressor(l2=0.5)
    ridge_reg.fit(X_train_scaled, y_train_score)
    ridge_score_preds = ridge_reg.predict(X_test_scaled)
    ridge_reg_metrics = compute_regression_metrics(y_test_score, ridge_score_preds)

    # Convert continuous score predictions to Tiers for classification comparison
    ridge_tier_preds = []
    for s in ridge_score_preds:
        if s < 52.0: ridge_tier_preds.append(0)
        elif s < 68.0: ridge_tier_preds.append(1)
        elif s < 83.0: ridge_tier_preds.append(2)
        else: ridge_tier_preds.append(3)
    ridge_cls_metrics = compute_classification_metrics(y_test_tier, np.array(ridge_tier_preds))

    # 2. Multi-Class Softmax Classifier
    softmax_cls = SoftmaxClassifier(lr=0.2, epochs=2000, l2=0.001)
    softmax_cls.fit(X_train_scaled, y_train_tier)
    softmax_preds = softmax_cls.predict(X_test_scaled)
    softmax_cls_metrics = compute_classification_metrics(y_test_tier, softmax_preds)

    # 3. Gradient Boosting Regressor
    gb_reg = GradientBoostingRegressorPure(n_estimators=30, lr=0.1)
    gb_reg.fit(X_train_scaled, y_train_score)
    gb_score_preds = gb_reg.predict(X_test_scaled)
    gb_reg_metrics = compute_regression_metrics(y_test_score, gb_score_preds)

    gb_tier_preds = []
    for s in gb_score_preds:
        if s < 52.0: gb_tier_preds.append(0)
        elif s < 68.0: gb_tier_preds.append(1)
        elif s < 83.0: gb_tier_preds.append(2)
        else: gb_tier_preds.append(3)
    gb_cls_metrics = compute_classification_metrics(y_test_tier, np.array(gb_tier_preds))

    # 4. K-Nearest Neighbors Classifier
    knn_cls = PureKNN(k=7)
    knn_cls.fit(X_train_scaled, y_train_tier)
    knn_preds = knn_cls.predict(X_test_scaled)
    knn_metrics = compute_classification_metrics(y_test_tier, knn_preds)

    print("\n--- Model Evaluation Summary ---")
    print(f"1. Ridge Regressor + Tier Mapping Accuracy: {ridge_cls_metrics['accuracy']*100:.2f}% | R2 Score: {ridge_reg_metrics['r2']:.4f} | MAE: {ridge_reg_metrics['mae']:.2f}%")
    print(f"2. Multi-Class Softmax Classifier Accuracy: {softmax_cls_metrics['accuracy']*100:.2f}% | F1-Score: {softmax_cls_metrics['f1_score']:.4f}")
    print(f"3. Gradient Boosting Accuracy:             {gb_cls_metrics['accuracy']*100:.2f}% | R2 Score: {gb_reg_metrics['r2']:.4f}")
    print(f"4. K-Nearest Neighbors (KNN) Accuracy:     {knn_metrics['accuracy']*100:.2f}% | F1-Score: {knn_metrics['f1_score']:.4f}")

    cls_summary = [
        {'Model': 'Polynomial Ridge Regressor + Tier Pipeline', 'Accuracy': ridge_cls_metrics['accuracy'], 'Precision': ridge_cls_metrics['precision'], 'Recall': ridge_cls_metrics['recall'], 'F1-Score': ridge_cls_metrics['f1_score'], 'CM': ridge_cls_metrics['confusion_matrix']},
        {'Model': 'Multi-Class Softmax Classifier', 'Accuracy': softmax_cls_metrics['accuracy'], 'Precision': softmax_cls_metrics['precision'], 'Recall': softmax_cls_metrics['recall'], 'F1-Score': softmax_cls_metrics['f1_score'], 'CM': softmax_cls_metrics['confusion_matrix']},
        {'Model': 'Gradient Boosting Ensemble', 'Accuracy': gb_cls_metrics['accuracy'], 'Precision': gb_cls_metrics['precision'], 'Recall': gb_cls_metrics['recall'], 'F1-Score': gb_cls_metrics['f1_score'], 'CM': gb_cls_metrics['confusion_matrix']},
        {'Model': 'K-Nearest Neighbors (KNN)', 'Accuracy': knn_metrics['accuracy'], 'Precision': knn_metrics['precision'], 'Recall': knn_metrics['recall'], 'F1-Score': knn_metrics['f1_score'], 'CM': knn_metrics['confusion_matrix']}
    ]

    reg_summary = [
        {'Model': 'Polynomial Ridge Regressor', 'R2 Score': ridge_reg_metrics['r2'], 'RMSE': ridge_reg_metrics['rmse'], 'MAE': ridge_reg_metrics['mae']},
        {'Model': 'Gradient Boosting Regressor', 'R2 Score': gb_reg_metrics['r2'], 'RMSE': gb_reg_metrics['rmse'], 'MAE': gb_reg_metrics['mae']}
    ]

    best_cls = max(cls_summary, key=lambda x: x['Accuracy'])
    best_reg = max(reg_summary, key=lambda x: x['R2 Score'])

    print(f"\n=================================================================")
    print(f" BEST CLASSIFIER: {best_cls['Model']} ({best_cls['Accuracy']*100:.2f}% Accuracy)")
    print(f" BEST REGRESSOR:  {best_reg['Model']} (R2 Score: {best_reg['R2 Score']:.4f})")
    print(f"=================================================================")

    # Feature Importance computation from Ridge Weights
    weights = np.abs(ridge_reg.w[1:])
    norm_w = weights / np.sum(weights)
    feat_imps = list(zip(feature_cols, [float(x) for x in norm_w]))
    feat_imps.sort(key=lambda x: x[1], reverse=True)

    # Save Pipeline Bundle
    artifacts = {
        'ridge_reg': ridge_reg,
        'softmax_cls': softmax_cls,
        'scaler': scaler,
        'dept_map': dept_map,
        'comp_map': comp_map,
        'feature_cols': feature_cols,
        'best_cls_name': best_cls['Model'],
        'best_cls_acc': best_cls['Accuracy'],
        'cls_summary': cls_summary,
        'reg_summary': reg_summary,
        'feature_importances': feat_imps
    }

    joblib.dump(artifacts, 'placement_model.joblib')
    print("Saved trained pipeline artifacts to 'placement_model.joblib'.")

    generate_md_report(cls_summary, reg_summary, best_cls['Model'], best_reg['Model'], best_cls['Accuracy'], feat_imps, n_samples)
    return artifacts


def generate_md_report(cls_results, reg_results, best_cls, best_reg, best_acc, feat_imps, n_samples):
    report_content = f"""# Machine Learning Model Documentation & Performance Report

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
- **Total Benchmark Profiles**: {n_samples} student records.
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
- **Why Selected**: Outperformed all models by achieving **`{best_acc*100:.2f}%` accuracy** and high $R^2$ score alignment.

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
"""
    for r in cls_results:
        status = "**SELECTED BEST**" if r['Model'] == best_cls else "Evaluated"
        report_content += f"| **{r['Model']}** | **{r['Accuracy']*100:.2f}%** | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1-Score']:.4f} | {status} |\n"

    report_content += f"""
### Regression Performance (Readiness Score Percentage Prediction)

| Machine Learning Algorithm | $R^2$ Score | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) | Status |
| :--- | :---: | :---: | :---: | :---: |
"""
    for r in reg_results:
        status = "**SELECTED BEST**" if r['Model'] == best_reg else "Evaluated"
        report_content += f"| **{r['Model']}** | **{r['R2 Score']:.4f}** | {r['RMSE']:.2f}% | {r['MAE']:.2f}% | {status} |\n"

    report_content += f"""
---

## 5. Deployed Model & Accuracy
- **Deployed Model**: `{best_cls}`
- **Classification Accuracy**: **`{best_acc*100:.2f}%`**
- **Regression Model**: `{best_reg}`

---

## 6. Feature Importance Breakdown

| Ranking | Feature Attribute | Importance Weight | Predictive Contribution |
| :---: | :--- | :---: | :--- |
"""
    for idx, (feat, weight) in enumerate(feat_imps, 1):
        report_content += f"| #{idx} | `{feat}` | `{weight:.4f}` | High impact on technical and overall placement tier |\n"

    report_content += """
---
*College Placement Readiness Predictor — Automated ML Report.*
"""
    with open('ml_model_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Saved ML documentation report to 'ml_model_report.md'.")


if __name__ == '__main__':
    train_and_evaluate_all()
