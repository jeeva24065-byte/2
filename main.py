import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('cleaned_data.csv')
X = df.drop(columns=['price'])
y_reg = df['price']
y_clf = (y_reg > y_reg.median()).astype(int)

ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
if 'education_level' in X.columns:
    X['education_level'] = X['education_level'].map(ordinal_mapping)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_ols = ols_model.predict(X_test_scaled)

ols_mse = mean_squared_error(y_reg_test, y_pred_reg_ols)
ols_r2 = r2_score(y_reg_test, y_pred_reg_ols)

print("=== OLS Linear Regression Coefficients ===")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': ols_model.coef_})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
print(coef_df[['Feature', 'Coefficient']])

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_ridge = ridge_model.predict(X_test_scaled)

ridge_mse = mean_squared_error(y_reg_test, y_pred_reg_ridge)
ridge_r2 = r2_score(y_reg_test, y_pred_reg_ridge)

print("\n=== Class Imbalance Check ===")
print("Before Resampling:\n", pd.Series(y_clf_train).value_counts())

minority_ratio = pd.Series(y_clf_train).value_counts(normalize=True).min()
if minority_ratio < 0.35:
    smote = SMOTE(random_state=42)
    X_train_clf_res, y_clf_train_res = smote.fit_resample(X_train_scaled, y_clf_train)
    print("After SMOTE Resampling:\n", pd.Series(y_clf_train_res).value_counts())
else:
    X_train_clf_res, y_clf_train_res = X_train_scaled, y_clf_train

clf_baseline = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf_baseline.fit(X_train_clf_res, y_clf_train_res)

y_pred_clf_base = clf_baseline.predict(X_test_scaled)
y_prob_clf_base = clf_baseline.predict_proba(X_test_scaled)[:, 1]

print("\n=== Baseline Classification Metrics ===")
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_pred_clf_base))
print("\nClassification Report:\n", classification_report(y_clf_test, y_pred_clf_base))

fpr_base, tpr_base, _ = roc_curve(y_clf_test, y_prob_clf_base)
auc_base = roc_auc_score(y_clf_test, y_prob_clf_base)

plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, label=f'Baseline C=1.0 (AUC = {auc_base:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

thresholds = np.arange(0.30, 0.71, 0.10)
thresh_results = []

for t in thresholds:
    t_preds = (y_prob_clf_base >= t).astype(int)
    p = precision_score(y_clf_test, t_preds, zero_division=0)
    r = recall_score(y_clf_test, t_preds, zero_division=0)
    f1 = f1_score(y_clf_test, t_preds, zero_division=0)
    thresh_results.append([f"{t:.2f}", f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

thresh_table = pd.DataFrame(thresh_results, columns=['Threshold', 'Precision', 'Recall', 'F1'])
print("\n=== Decision Threshold Sensitivity Table ===")
print(thresh_table.to_string(index=False))

clf_strong_reg = LogisticRegression(max_iter=1000, C=0.01, random_state=42)
clf_strong_reg.fit(X_train_clf_res, y_clf_train_res)

y_pred_clf_reg = clf_strong_reg.predict(X_test_scaled)
y_prob_clf_reg = clf_strong_reg.predict_proba(X_test_scaled)[:, 1]

p_reg = precision_score(y_clf_test, y_pred_clf_reg, zero_division=0)
r_reg = recall_score(y_clf_test, y_pred_clf_reg, zero_division=0)
auc_reg = roc_auc_score(y_clf_test, y_prob_clf_reg)

n_bootstraps = 500
bootstrapped_auc_diffs = []
y_clf_test_arr = np.array(y_clf_test)

for _ in range(n_bootstraps):
    indices = np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)
    if len(np.unique(y_clf_test_arr[indices])) < 2:
        continue
    auc_base_boot = roc_auc_score(y_clf_test_arr[indices], y_prob_clf_base[indices])
    auc_reg_boot = roc_auc_score(y_clf_test_arr[indices], y_prob_clf_reg[indices])
    bootstrapped_auc_diffs.append(auc_base_boot - auc_reg_boot)

mean_diff = np.mean(bootstrapped_auc_diffs)
ci_lower = np.percentile(bootstrapped_auc_diffs, 2.5)
ci_upper = np.percentile(bootstrapped_auc_diffs, 97.5)

print("\n=== Bootstrap Results ===")
print(f"Mean AUC Difference: {mean_diff:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
