# Credit Scoring Model
**CodeAlpha Machine Learning Internship – Task 1**

A machine learning project that predicts an individual's creditworthiness (Good vs. Bad credit) from financial history and compares three classification algorithms.

## Dataset
- **Source:** German Credit Dataset (`credit-g` via `sklearn.datasets.fetch_openml`)
- **Size:** 1,000 samples (700 Good Credit, 300 Bad Credit), 20 original features
- **Feature Engineering:** Added `credit_per_month = credit_amount / duration`

## Approach
1. Stratified 80/20 train-test split (`stratify=y`) to preserve class balance.
2. Preprocessing inside a `scikit-learn` `Pipeline` (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features) to prevent data leakage.
3. Handled class imbalance using `class_weight='balanced'`.
4. Compared **Logistic Regression**, **Decision Tree**, and **Random Forest**.

## Results
Evaluated on the 200-sample test set (140 Good, 60 Bad):

| Model | ROC-AUC | Accuracy | Recall (Bad Credit) | Precision (Bad Credit) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.772** | **74%** | 0.35 | **0.62** |
| **Logistic Regression** | 0.762 | 68% | **0.70** | 0.47 |
| **Decision Tree** | 0.629 | 64% | 0.58 | 0.42 |

## Key Takeaways
Random Forest achieved the highest ROC-AUC (0.772) and accuracy (74%), but it missed 39 of 60 bad-credit customers (Recall = 0.35). Logistic Regression, with a similar AUC (0.762), caught 70% (42 of 60) of bad-credit customers, at the cost of rejecting more good customers (47 vs. 13). The best choice depends on the bank's priority: minimizing loan defaults or minimizing lost good customers.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## How to Run
1. Open `Credit_Scoring_Model.ipynb` in Google Colab.
2. Run all cells (`Runtime → Run all`). The dataset loads automatically via OpenML.
