# Part 2: Supervised Machine Learning Model Documentation

## 1. Label Definitions & Preprocessing Decisions
* Feature Matrix ($X$): Consists of all variables from `cleaned_data.csv` excluding the target column (`price`).
* Regression Label ($y_{reg}$): Continuous values from the `price` column.
* Classification Label ($y_{clf}$): Generated using a binary rule where $y_{clf} = 1$ if the entry is greater than the target median value, and $0$ otherwise. 

### Categorical Encoding Strategies
* Ordinal Encoding: Applied directly to `education_level` mapped as `{'Low': 0, 'Medium': 1, 'High': 2}`. This naturally preserves structural data context hierarchy (Low < Medium < High).
* One-Hot Encoding: Applied to `city_name` using `pd.get_dummies()`, dropping the first dummy column. One-hot encoding avoids introducing artificial mathematical distance or sequence scaling rules where no intrinsic hierarchy exists.

### Data Leakage Guardrails
The `StandardScaler` was fit exclusively on the training features (`scaler.fit(X_train)`) and applied downstream via `.transform()`. Fitting a feature scaler globally across the entire collection introduces direct data leakage by calculating properties (global standard deviation and sample means) from testing distributions. This falsely deflates performance tracking errors during validation.

---

## 2. Regression Model Evaluation

### Metrics Comparison Table
| Model Variant | Mean Squared Error (MSE) | R² Score |
| :--- | :--- | :--- |
| OLS Linear Regression | 2582.41 | -0.014 |
| Ridge Regression ($\alpha = 1.0$) | 2582.39 | -0.014 |

### Coefficient Profile Interpretations
* Large Positive Coefficients: For every individual unit step up in the scaled feature variation metric, the system projects an explicit baseline target parameter step growth proportional to the raw coefficient calculation weight.
* Large Negative Coefficients: Higher scaled input attributes scale baseline response factors down linearly against the calculated inverse feature multiplier.
* OLS vs. Ridge Coefficients: Ridge Regression applies an $L_2$ regularization penalty that shrinks coefficients toward zero. The $\alpha$ parameter directly controls this trade-off: a higher $\alpha$ penalizes large coefficients more heavily to decrease model variance, whereas an $\alpha = 0$ yields ordinary least squares (OLS) linear regression.

---

## 3. Classification Model Evaluation

### Precision and Recall Formulas
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Balancing Class Constraints
The baseline distribution showed a severe minority class imbalance (<35% representation). SMOTE (Synthetic Minority Over-sampling Technique) was introduced exclusively on the training sets to balance structural visibility rules safely, preventing the classifier from developing an algorithmic bias toward the majority population.

### ROC/AUC Baseline Performance Meaning
The Area Under the ROC Curve (AUC) calculates the exact probability that the trained model will score a randomly selected true positive sample higher than a randomly selected true negative one. An AUC closer to 1.0 indicates near-perfect class separation capability.

---

## 4. Decision-Threshold Sensitivity Analysis

### Strategic Metrics Matrix
| Threshold | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| 0.30 | 0.231 | 0.429 | 0.300 |
| 0.40 | 0.250 | 0.286 | 0.267 |
| 0.50 | 0.222 | 0.143 | 0.174 |
| 0.60 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0.000 | 0.000 | 0.000 |

### Optimization & Domain Logic
* F1 Optimization Threshold: The highest $F_1$-score on this dataset is achieved at a threshold of 0.30.
* Domain Priority Selection: If false negatives are more costly, Recall becomes the primary optimization metric. To optimize for Recall, the decision threshold must be lowered. The cost of this adjustment is an increase in False Positives, which consequently drives down the model's overall Precision.

---

## 5. Regularization Experiment Matrix

### Hyperparameter Shift Results
| Model Configuration | Precision | Recall | AUC Score |
| :--- | :--- | :--- | :--- |
| Baseline ($C = 1.0$) | 0.222 | 0.143 | 0.490 |
| Strong Regularization ($C = 0.01$) | 0.000 | 0.000 | 0.472 |

The parameter $C$ controls the inverse regularization strength in Logistic Regression; smaller values impose a more severe L2 penalty to aggressively flatten feature weights. On this dataset, reducing $C$ down to $0.01$ restricted model flexibility too heavily, collapsing predictive yields entirely down to zero.

### Bootstrap Variance Confidence Limits
* Mean AUC Difference: 0.0182
* 95% Percentile Confidence Range: [-0.0814, 0.1250]

Conclusion: Because the 95% Bootstrap Confidence Interval includes zero, the performance advantage of the $C=1.0$ baseline model over the heavily regularized $C=0.01$ configuration is not statistically reliable across resampled test slices.
