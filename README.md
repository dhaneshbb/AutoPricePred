<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

# Automobile Price Prediction

Machine learning regression model for predicting 1985 import vehicle prices. Achieves 91.7% test R-squared and 89.4% cross-validation R-squared with Lasso regression.

</div>

---

## Overview

Predicts automobile prices from 25 vehicle specification features (brand, engine, body design, performance). Addresses data leakage, missing values (18%), multicollinearity (VIF > 16,000), outliers (10% of samples), and high-cardinality categoricals.

**Final Model:** Lasso Regression (alpha=10.0)

- Test: R-squared = 0.917, RMSE = $1,987
- Cross-validation: R-squared = 0.894 +/- 0.027
- Features: 42 (6 PCA + 36 categorical), 29 non-zero coefficients
- Overfitting gap: 3.3%

See [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) for why Lasso was selected over XGBoost despite 16% higher test error.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Load Trained Model

```python
import joblib
model = joblib.load('models/final_lasso_model.joblib')
predictions = model.predict(X_new)  # Requires 42 engineered features
```

### Run Full Pipeline

Open `notebooks/auto-price-prediction.ipynb` for complete data preparation, modeling, and evaluation workflow.

## Dataset

| Property | Details |
|----------|---------|
| **Source** | 1985 Auto Imports Database (UCI ML Repository) |
| **Samples** | 200 vehicles (205 original, 5 removed) |
| **Features** | 26 attributes (25 predictors + price) |
| **Split** | 158 train / 40 test (stratified) |

See [data/raw/dataset-info.txt](data/raw/dataset-info.txt) for full metadata.

## Project Structure

**Core directories:**
- `data/raw/` - Original dataset (auto_imports.csv)
- `data/processed/train-test/` - Train/test split
- `notebooks/` - Full analysis pipeline (auto-price-prediction.ipynb)
- `src/` - Reusable modules (utils, statistical analysis, model evaluation)
- `models/` - Trained model artifacts (final_lasso_model.joblib)
- `reports/` - Detailed analysis reports

## Working with the Notebook

**Import pattern used:**
The notebook imports functions from src/ modules using:
```python
from src.utils import memory_usage, dataframe_memory_usage
from src.statistical_analysis import normality_test_with_skew_kurt, spearman_correlation_with_target
from src.model_evaluation import evaluate_regression_model, hyperparameter_tuning
```

**Running analysis:**
The notebook contains the full ML pipeline. Execute cells sequentially for:
1. Data loading and cleaning
2. Statistical analysis (normality tests, correlation)
3. Feature engineering (PCA on numerical features)
4. Model comparison (10 algorithms tested)
5. Hyperparameter tuning (5-fold GridSearchCV)
6. Final model selection and persistence

## Model Training Workflow

**Base model evaluation:**
```python
from src.model_evaluation import evaluate_regression_model

metrics = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
# Returns: MAE, MSE, RMSE, R², Adjusted R², MSLE, MAPE, CV R², Training R², Overfit, Training Time
```

**Hyperparameter tuning:**
```python
from src.model_evaluation import hyperparameter_tuning

best_models, best_params, times = hyperparameter_tuning(
    models={'Lasso': Lasso()},
    param_grids={'Lasso': {'alpha': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]}},
    X_train=X_train,
    y_train=y_train,
    scoring_metric='neg_mean_squared_error',
    cv_folds=5
)
```

## Statistical Analysis Functions

**Normality testing:**
```python
from src.statistical_analysis import normality_test_with_skew_kurt

normal_df, not_normal_df = normality_test_with_skew_kurt(df)
# Uses Shapiro-Wilk (n<=5000) or Kolmogorov-Smirnov (n>5000)
```

**Multicollinearity detection:**
```python
from src.statistical_analysis import calculate_vif

vif_data, high_vif_features = calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0)
# Returns VIF scores and features exceeding threshold
```

**Spearman correlation:**
```python
from src.statistical_analysis import spearman_correlation_with_target

corr_data = spearman_correlation_with_target(
    data,
    non_normal_cols=['col1', 'col2'],
    target_col='TARGET',
    plot=True,
    table=True
)
```

## Model Persistence

**Loading the final model:**
```python
import joblib
model = joblib.load('models/final_lasso_model.joblib')
predictions = model.predict(X_new)
```

## Key Design Decisions

**Model selection criteria (weighted):**
1. Generalization (CV R² stability) - 40%
2. Accuracy (Test RMSE/R²) - 30%
3. Stability (overfitting gap) - 20%
4. Efficiency (training time, interpretability) - 10%

**Why Lasso over XGBoost:**
- XGBoost achieved lower test RMSE (1,663 vs 1,987) but showed 8.3-point CV-test gap vs Lasso's 2.3-point gap
- Lasso provides interpretability (29 sparse coefficients) vs XGBoost black box
- Training: 11.5x faster (0.014s vs 0.161s)
- Inference: 600x faster operations
- Trade-off: Accept 2.5% higher error for 4.1 points better CV R² and full transparency

## Reports

Detailed analysis in `reports/`:

- [Complete_Data_Analysis_Report.md](reports/Complete_Data_Analysis_Report.md) - Full methodology, statistical analysis, and results
- [Model_Comparison_Report.md](reports/Model_Comparison_Report.md) - Model selection rationale and performance comparison
- [Challenges_Report.md](reports/Challenges_Report.md) - Technical challenges and solutions
- [GALLERY.md](results/figures/GALLERY.md) - Visualizations

## Development

### Code Quality

```bash
# Format code
black .
isort .

# Run pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

- black (88-char lines)
- isort (black-compatible)
- nbqa-black (notebooks)
- Validation (YAML, JSON, trailing whitespace)

---


- MIT License - Copyright (c) 2025 Dhanesh B. B.
- GitHub: [https://github.com/dhaneshbb](https://github.com/dhaneshbb)
