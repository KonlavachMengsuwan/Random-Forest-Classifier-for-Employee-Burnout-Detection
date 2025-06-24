# 📊 Random Forest Classifier for Employee Burnout Detection

This project uses a **Random Forest Classifier** to predict employee burnout based on synthetic workplace data. It includes **data preprocessing**, **model training**, **hyperparameter tuning**, and **SHAP explainability**.
![image](https://github.com/user-attachments/assets/4805d2ae-daac-4186-90f9-06a76d099143)

---

## 🧠 Motivation

Burnout is a growing concern in modern workplaces. Early detection using machine learning could help mitigate negative impacts on employees and organizations. This project explores:

* Can a simple Random Forest predict burnout reliably?
* What features most influence burnout prediction?
* How do different Random Forest hyperparameters affect model performance?
* Can SHAP help explain individual predictions?

---

## 🔍 Dataset

A synthetic dataset (`synthetic_employee_burnout.csv`) simulates employee data with these features:

* `Name` (dropped)
* `Gender` (categorical)
* `JobRole` (categorical)
* `Age`, `Experience`, `RemoteRatio`, `WorkHoursPerWeek`, `SatisfactionLevel`, `StressLevel`
* Target: `Burnout` (0 = No, 1 = Yes)

---

## 🚀 Objectives

* Train a Random Forest model to classify burnout.
* Tune hyperparameters using `GridSearchCV`.
* Analyze model performance.
* Use SHAP to explain predictions.

---

## 🧪 Methods

### ✅ 1. Install and Import Libraries

```python
!pip install pandas numpy matplotlib seaborn scikit-learn -q
import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
```

### 📂 2. Load and Preprocess Data

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('synthetic_employee_burnout.csv')
df = df.drop(columns=['Name'])

# Encode categorical
for col in ['Gender', 'JobRole']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Burnout', axis=1)
y = df['Burnout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 🌳 3. Train Initial Random Forest

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 📈 4. Evaluate Model

```python
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 📊 5. Feature Importance

```python
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
```

### 📊 Classification Report

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| **0**        | 1.00      | 1.00   | 1.00     | 374     |
| **1**        | 1.00      | 1.00   | 1.00     | 26      |
|              |           |        |          |         |
| **Accuracy** |           |        | **1.00** | 400     |
| **Macro avg**| 1.00      | 1.00   | 1.00     | 400     |
| **Weighted avg** | 1.00  | 1.00   | 1.00     | 400     |


![image](https://github.com/user-attachments/assets/6e21badf-0f31-4118-a6d7-d40527fe635d)


---

## 🛠️ Hyperparameter Tuning (Grid Search)

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### 🔍 Best Parameters

```python
print(grid_search.best_params_)
```

### 📊 Evaluate Best Model

```python
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
```

---

## 📉 Visualizing Hyperparameter Impact

```python
results_df = pd.DataFrame(grid_search.cv_results_)

# 1D Plots
sns.lineplot(x='param_n_estimators', y='mean_test_score', data=results_df)
plt.title("Effect of n_estimators")
plt.show()

sns.lineplot(x='param_max_depth', y='mean_test_score', data=results_df)
plt.title("Effect of max_depth")
plt.show()

sns.lineplot(x='param_min_samples_split', y='mean_test_score', data=results_df)
plt.title("Effect of min_samples_split")
plt.show()

sns.lineplot(x='param_min_samples_leaf', y='mean_test_score', data=results_df)
plt.title("Effect of min_samples_leaf")
plt.show()
```

### 🔥 2D Heatmaps

```python
results_df['param_max_depth'] = results_df['param_max_depth'].astype(str)

# n_estimators vs max_depth
sns.heatmap(results_df.pivot('param_n_estimators', 'param_max_depth', 'mean_test_score'), annot=True, fmt='.3f')
plt.title("Accuracy: n_estimators vs max_depth")
plt.show()

# min_samples_split vs min_samples_leaf
sns.heatmap(results_df.pivot('param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score'), annot=True, fmt='.3f', cmap="magma")
plt.title("Accuracy: min_samples_split vs min_samples_leaf")
plt.show()
```

---

## 🧠 Explainable AI (SHAP)

```python
!pip install shap==0.41.0 -q
import shap
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

# Beeswarm Plot
shap.plots.beeswarm(shap_values[..., 1], max_display=10)

# Waterfall for one person
shap.plots.waterfall(shap_values[..., 1][0])

# Force plot
shap.initjs()
shap.plots.force(shap_values[..., 1][0], matplotlib=True)
```

---

## 🔧 Hyperparameter Meaning

| Parameter           | Description                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| `n_estimators`      | Number of decision trees. Higher = more stable but slower.                   |
| `max_depth`         | Max depth of each tree. None = full growth. Lower depth reduces overfitting. |
| `min_samples_split` | Minimum samples to split a node. Higher = more conservative trees.           |
| `min_samples_leaf`  | Minimum samples in a leaf. Helps smooth noisy data.                          |

---

## ✅ Summary

* **Random Forest** predicts burnout with very high accuracy.
* **Key Features**: Stress level, satisfaction, work hours.
* **SHAP** provides transparent model interpretation.
* **GridSearchCV** fine-tunes performance and improves generalization.

---

## 📁 Files

* `synthetic_employee_burnout.csv`: synthetic dataset
* `notebook.ipynb`: Colab-compatible training + tuning + SHAP analysis script

---

## 👤 Author

Built by \[Your Name] as an educational project exploring machine learning and explainability using Random Forest and SHAP.

---

Let me know if you'd like me to help convert this into Markdown or push to a real GitHub repo!
