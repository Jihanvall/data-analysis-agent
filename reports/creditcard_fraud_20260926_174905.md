# Data Analysis Report: creditcard_fraud

## Preprocessing Plan

- **Time** → `drop_column`: High unique values suggest an identifier; not suitable for direct use.
- **Class** → `none`: Is the target column for classification task.

## Applied Steps Log

- learned drop_column on 'Time'
- applied drop_column on 'Time'
- applied drop_column on 'Time'

## Model Selection

**Best model:** `random_forest`

Cross-validation scores:

- logistic_regression: 0.9949
- random_forest: 0.9958 ✅
- gradient_boosting: 0.9936

## Test Set Performance

- **f1_weighted**: 0.9948
- **accuracy**: 0.995
