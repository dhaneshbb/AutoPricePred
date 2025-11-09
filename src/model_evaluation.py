import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    msle = mean_squared_log_error(y_test, y_pred) if np.all(y_pred >= 0) else None
    mape = (
        mean_absolute_percentage_error(y_test, y_pred) if np.all(y_pred >= 0) else None
    )
    r2_train = r2_score(y_train, y_pred_train)
    cv_r2 = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="r2"))
    overfit = r2_train - r2
    return {
        "Model Name": type(model).__name__,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adjusted_r2,
        "MSLE": msle,
        "MAPE": mape,
        "Cross-Validation R2": cv_r2,
        "Training R2": r2_train,
        "Overfit": overfit,
        "Training Time (seconds)": round(training_time, 4),
    }


def visualize_model_performance(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Model Performance: {type(model).__name__}", fontsize=14)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=3, scoring="r2"
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    axes[0, 0].plot(train_sizes, train_mean, "o-", label="Train Score")
    axes[0, 0].plot(train_sizes, test_mean, "o-", label="Test Score")
    axes[0, 0].set_title("Learning Curve")
    axes[0, 0].set_xlabel("Training Samples")
    axes[0, 0].set_ylabel("R2 Score")
    axes[0, 0].legend()
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, color="blue")
    axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--r")
    axes[0, 1].set_title("True vs Predicted (Test)")
    axes[0, 1].set_xlabel("True Values")
    axes[0, 1].set_ylabel("Predicted Values")
    axes[0, 2].scatter(y_pred_test, residuals_test, alpha=0.5, color="purple")
    axes[0, 2].axhline(y=0, color="r", linestyle="--")
    axes[0, 2].set_title("Residuals vs Predicted (Test)")
    axes[0, 2].set_xlabel("Predicted Values")
    axes[0, 2].set_ylabel("Residuals")
    sns.histplot(residuals_test, bins=30, kde=True, ax=axes[1, 0], color="teal")
    axes[1, 0].set_title("Test Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals")
    sns.histplot(
        residuals_train, bins=30, kde=True, ax=axes[1, 1], color="green", alpha=0.7
    )
    axes[1, 1].set_title("Train Residuals Distribution")
    axes[1, 1].set_xlabel("Residuals")
    stats.probplot(residuals_test, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title("QQ Plot (Test Residuals)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def hyperparameter_tuning(
    models,
    param_grids,
    X_train,
    y_train,
    scoring_metric="neg_mean_squared_error",
    cv_folds=5,
):
    best_models = {}
    best_params = {}
    execution_times = {}
    for model_name, model in models.items():
        print(f"Starting grid search for {model_name}...")
        start_time = time.time()
        if model_name in param_grids:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                scoring=scoring_metric,
                cv=cv_folds,
                verbose=1,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            best_params[model_name] = grid_search.best_params_
            execution_times[model_name] = time.time() - start_time
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(
                f"Completed grid search for {model_name} in {execution_times[model_name]:.2f} seconds.\n"
            )
        else:
            print(f"No parameter grid available for {model_name}.")
    return best_models, best_params, execution_times
