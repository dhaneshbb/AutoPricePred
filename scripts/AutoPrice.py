import pandas as pd
data = pd.read_csv(r'D:\datamites\AutoPricePred\data\1.1 raw\auto_imports.csv')

from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

from insightfulpy.eda import *

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, pearsonr, spearmanr,
    ttest_ind, mannwhitneyu, shapiro
)
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, mean_squared_log_error
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA

import joblib


import psutil
import os
import gc

def memory_usage():
    """Prints the current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def dataframe_memory_usage(df):
    """Returns the memory usage of a Pandas DataFrame in MB."""
    mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"DataFrame Memory Usage: {mem_usage:.2f} MB")
    return mem_usage

def garbage_collection():
    """Performs garbage collection to free up memory."""
    gc.collect()
    memory_usage()

def normality_test_with_skew_kurt(df):
    normal_cols = []
    not_normal_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) >= 3:
            if len(col_data) <= 5000:
                stat, p_value = shapiro(col_data)
                test_used = 'Shapiro-Wilk'
            else:
                stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_used = 'Kolmogorov-Smirnov'
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            result = {
                'Column': col,
                'Test': test_used,
                'Statistic': stat,
                'p_value': p_value,
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
            if p_value > 0.05:
                normal_cols.append(result)
            else:
                not_normal_cols.append(result)
    normal_df = (
        pd.DataFrame(normal_cols)
        .sort_values(by='Column') 
        if normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols)
        .sort_values(by='p_value', ascending=False)  # Sort by p-value descending (near normal to not normal)
        if not_normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p ≤ 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df
def spearman_correlation_with_target(data, non_normal_cols, target_col='TARGET', plot=True, table=True):
    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric. Please encode it before running this test.")
    correlation_results = {}
    for col in non_normal_cols:
        if col not in data.columns:
            continue 
        coef, p_value = spearmanr(data[col], data[target_col], nan_policy='omit')
        correlation_results[col] = {'Spearman Coefficient': coef, 'p-value': p_value}
    correlation_data = pd.DataFrame(correlation_results).T.dropna()
    correlation_data = correlation_data.sort_values('Spearman Coefficient', ascending=False)
    if target_col in correlation_data.index:
        correlation_data = correlation_data.drop(target_col)
    positive_corr = correlation_data[correlation_data['Spearman Coefficient'] > 0]
    negative_corr = correlation_data[correlation_data['Spearman Coefficient'] < 0]
    if table:
        print(f"\nPositive Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in positive_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
        print(f"\nNegative Spearman Correlations with Target ('{target_col}'):\n")
        for feature, stats in negative_corr.iterrows():
            print(f"- {feature}: Correlation={stats['Spearman Coefficient']:.4f}, p-value={stats['p-value']:.4f}")
    if plot:
        plt.figure(figsize=(20, 8))  # Increase figure width to prevent label overlap
        sns.barplot(x=correlation_data.index, y='Spearman Coefficient', data=correlation_data, palette='coolwarm')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"Spearman Correlation with Target ('{target_col}')", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Spearman Coefficient", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels for clarity
        plt.subplots_adjust(bottom=0.3)  # Add space below the plot for labels
        plt.tight_layout()
        plt.show()
    return correlation_data
def spearman_correlation(data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8):
    if non_normal_cols.empty:
        print("\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation.")
        return
    selected_columns = non_normal_cols['Column'].tolist()
    if exclude_target and exclude_target in selected_columns and pd.api.types.is_numeric_dtype(data[exclude_target]):
        selected_columns.remove(exclude_target)
    spearman_corr_matrix = data[selected_columns].corr(method='spearman')
    multicollinear_pairs = []
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i+1:]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    print("\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(multicollinearity_threshold))
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Keep reasonable bounds
    annot_font_size = max(min(10, 200 / num_vars), 6)  # Smaller font for more variables
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()
def calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=[exclude_target], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_data.values, i) 
                       for i in range(numeric_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data['VIF'] > multicollinearity_threshold]
    low_vif = vif_data[vif_data['VIF'] <= multicollinearity_threshold]
    print(f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):")
    print("\nFeatures with VIF > threshold (High Multicollinearity):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("None. No features exceed the VIF threshold.")
    print("\nFeatures with VIF <= threshold (Low/No Multicollinearity):")
    if not low_vif.empty:
        print(low_vif.to_string(index=False))
    else:
        print("None. All features exceed the VIF threshold.")
    return vif_data, high_vif['Feature'].tolist()

if __name__ == "__main__":
    memory_usage()

dataframe_memory_usage(data)

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

# # # Launch D-Tale
# d = dtale.show(data)
# d.open_browser()

column_names = [
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
    'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 
    'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 
    'price'
]
data.columns = column_names

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

detect_mixed_data_types(data)

cat_high_cardinality(data)

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates

inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

data.dtypes.value_counts()

columns_info("Dataset Overview", data)
# analyze_data(data)

data.replace('?', np.nan, inplace=True)

data["normalized-losses"] = pd.to_numeric(data["normalized-losses"], errors="coerce").astype("float64")
data["bore"] = pd.to_numeric(data["bore"], errors="coerce").astype("float64")
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce").astype("float64")
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data["peak-rpm"] = pd.to_numeric(data["peak-rpm"], errors="coerce")

data["horsepower"].fillna(data["horsepower"].median(), inplace=True)
data["peak-rpm"].fillna(data["peak-rpm"].median(), inplace=True)
data["horsepower"] = data["horsepower"].astype("int64")
data["peak-rpm"] = data["peak-rpm"].astype("int64")

cat_cols = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"
]
data[cat_cols] = data[cat_cols].astype("category")

missing_inf_values(data)
show_missing(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates

data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
data_missing_summary, data_non_missing_summary = comp_num_analysis(data, missing_df=True)
data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_cat_missing_summary.shape)
print(data_missing_summary.shape)
print(data_outlier_summary.shape)

data_cat_missing_summary

data_missing_summary

categorical_cols = ["num-of-doors"]
for col in categorical_cols:
    mode_value = data[col].mode()[0]  
    data[col].fillna(mode_value, inplace=True)

num_cols_missing = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]
for col in num_cols_missing:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)

plot_boxplots(data)
calculate_skewness_kurtosis(data)

data_outlier_summary

cap_features = ["city-mpg", "highway-mpg", "engine-size", "peak-rpm",
              "horsepower", "stroke", "compression-ratio"]
for col in cap_features:
    upper_bound = data[col].quantile(0.99) 
    if data[col].dtype == 'int64':  
        data[col] = np.where(data[col] > upper_bound, int(upper_bound), data[col])
    else:  
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

realistic_thresholds = {
    "compression-ratio": (7, 15),  
    "stroke": (2.0, 4.0),  
    "price": (5000, 100000), 
    "width": (60, 80), 
    "engine-size": (50, 500),  
    "horsepower": (40, 1000),  
    "normalized-losses": (50, 250), 
    "highway-mpg": (10, 80), 
    "wheel-base": (80, 140),  
    "length": (120, 250),
}
for col, (lower, upper) in realistic_thresholds.items():
    unrealistic_values = data[(data[col] < lower) | (data[col] > upper)][col]
    if not unrealistic_values.empty:
        print(f"Unrealistic values found in '{col}':")
        print(unrealistic_values)
unrealistic_counts = {
    col: data[(data[col] < lower) | (data[col] > upper)][col].count()
    for col, (lower, upper) in realistic_thresholds.items()
}
print("\nSummary of Unrealistic Values:")
print(unrealistic_counts)

compression_ratio_upper_bound = 15
data["compression-ratio"] = np.where(data["compression-ratio"] > compression_ratio_upper_bound,
                                     compression_ratio_upper_bound, 
                                     data["compression-ratio"])

outlier_cols = ["compression-ratio", "stroke", "price", "width", "engine-size",
                "horsepower", "highway-mpg", "wheel-base", "length"]
interconnected_outliers_df = interconnected_outliers(data, outlier_cols)

interconnected_outliers_df

interconnected_cols = ["compression-ratio", "stroke", "width", "engine-size", 
                       "horsepower", "wheel-base", "highway-mpg", "length"]

for col in interconnected_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR  
    if data[col].dtype == "int64":
        data[col] = np.where(data[col] > upper_bound, int(upper_bound), data[col])
        data[col] = data[col].astype("int64")  # Ensure integer type remains
    else:
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
data.loc[data["engine-size"] > data["engine-size"].quantile(0.99), "engine-size"] = int(data["engine-size"].median())
data.loc[data["horsepower"] > data["horsepower"].quantile(0.99), "horsepower"] = int(data["horsepower"].median())

iqr_cap_cols = ["stroke", "compression-ratio", "length", "width",]
for col in iqr_cap_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if data[col].dtype == 'int64':
        data[col] = np.where(data[col] > upper_bound, int(upper_bound),
                             np.where(data[col] < lower_bound, int(lower_bound), data[col]))
        data[col] = data[col].astype("int64")  # Ensure int type remains
    else:
        data[col] = np.where(data[col] > upper_bound, upper_bound,
                             np.where(data[col] < lower_bound, lower_bound, data[col]))
remaining_outliers = data[(data[iqr_cap_cols] > data[iqr_cap_cols].quantile(0.75) + 1.5 * (data[iqr_cap_cols].quantile(0.75) - data[iqr_cap_cols].quantile(0.25))) | 
                          (data[iqr_cap_cols] < data[iqr_cap_cols].quantile(0.25) - 1.5 * (data[iqr_cap_cols].quantile(0.75) - data[iqr_cap_cols].quantile(0.25)))]


upper_limit = 29589.375
data['price'] = data['price'].apply(lambda x: min(x, upper_limit))

data_negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()
data_negative_values = data_negative_values[data_negative_values > 0].sort_values(ascending=False)
print("Negative Values (Sorted):\n", data_negative_values)

correlation = data[['normalized-losses', 'price']].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Between Normalized-Losses and Price")
plt.show()

data = data.drop(columns=["normalized-losses"], errors="ignore")

num_analysis_and_plot(data, 'price')

columns_info("Dataset Overview", data)

analyze_data(data)

num_summary(data)

cat_summary(data)

kde_batches(data, batch_num=1)
kde_batches(data, batch_num=2)
box_plot_batches(data, batch_num=1)
box_plot_batches(data, batch_num=2)
qq_plot_batches(data, batch_num=1)
qq_plot_batches(data, batch_num=2)

cat_bar_batches(data, batch_num=1,high_cardinality_limit=22)
cat_pie_chart_batches(data, batch_num=1,high_cardinality_limit=22)

features = ["engine-size", "curb-weight", "horsepower", "highway-mpg"]
# Create scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, feature in enumerate(features):
    axes[i].scatter(data[feature], data["price"], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Price")
    axes[i].set_title(f"Price vs {feature}")
plt.tight_layout()
plt.show()

num_vs_cat_box_violin_pair_batch(data, pair_num=14, batch_num=1, high_cardinality_limit=22)

num_vs_num_scatterplot_pair_batch(data, pair_num=14, batch_num=1, hue_column="price")
num_vs_num_scatterplot_pair_batch(data, pair_num=14, batch_num=2, hue_column="price")

cat_vs_cat_pair_batch(data, pair_num=1, batch_num=1, high_cardinality_limit=22)
cat_vs_cat_pair_batch(data, pair_num=5, batch_num=1, high_cardinality_limit=22)
cat_vs_cat_pair_batch(data, pair_num=7, batch_num=1, high_cardinality_limit=22)
cat_vs_cat_pair_batch(data, pair_num=8, batch_num=1, high_cardinality_limit=22)

data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
data_outlier_summary

data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
data_cat_non_missing_summary

categorical_features = [
    'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
    'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system'
]
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(data[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
data = pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates

data.drop_duplicates(inplace=True)

retain_features = [
    'width', 'compression-ratio', 'highway-mpg', 'curb-weight', 'engine-size',
    'horsepower', 'city-mpg', 'body-style_sedan', 'drive-wheels_fwd', 'drive-wheels_rwd'
]
features = data.drop(columns=['price'])  # Exclude target variable
vif_data = pd.DataFrame()
vif_data["Feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
high_vif_threshold = 8.0
removable_features = vif_data[(vif_data["VIF"] > high_vif_threshold) & (~vif_data["Feature"].isin(retain_features))]
print("\n **Initial VIF Analysis Before Removal:**")
print(vif_data.sort_values(by="VIF", ascending=False))
while not removable_features.empty:
    drop_feature = removable_features.sort_values("VIF", ascending=False).iloc[0]["Feature"]
    features.drop(columns=[drop_feature], inplace=True)
    print(f" Dropping '{drop_feature}' (VIF={removable_features.loc[removable_features['Feature'] == drop_feature, 'VIF'].values[0]:.2f}) due to high multicollinearity.")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    removable_features = vif_data[(vif_data["VIF"] > high_vif_threshold) & (~vif_data["Feature"].isin(retain_features))]
print("\n **Final VIF Analysis After Cleanup:**")
print(vif_data.sort_values(by="VIF", ascending=False))
data = pd.concat([features, data[['price']]], axis=1)
print("\n after preventing over-removal:", data.shape)

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)
spearman_correlation(data, data_not_normal_df, exclude_target='price', multicollinearity_threshold=0.8)

above_threshold, below_threshold = calculate_vif(data, exclude_target='price', multicollinearity_threshold=8.0)

X = data.drop(columns=['price'])  
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[retain_features])
X_test_scaled = scaler.transform(X_test[retain_features])

pca = PCA(n_components=len(retain_features)) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.argmax(explained_variance >= 0.95) + 1 

pca = PCA(n_components=optimal_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_columns = [f"PCA_{i+1}" for i in range(pca.n_components_)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)
X_train_final = pd.concat([X_train.drop(columns=retain_features), X_train_pca_df], axis=1)
X_test_final = pd.concat([X_test.drop(columns=retain_features), X_test_pca_df], axis=1)
print("\n PCA Applied to Retained Features.")
print(f" Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f" Final Train Set Shape: {X_train_final.shape}, Test Set Shape: {X_test_final.shape}")

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_final.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_final.values, i) for i in range(X_train_final.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False))

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
    mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(y_pred >= 0) else None
    r2_train = r2_score(y_train, y_pred_train)
    cv_r2 = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring="r2"))
    overfit = r2_train - r2
    return {
        "Model Name": type(model).__name__,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2,
        "MSLE": msle,
        "MAPE": mape,
        "Cross-Validation R²": cv_r2,
        "Training R²": r2_train,
        "Overfit": overfit,
        "Training Time (seconds)": round(training_time, 4)
    }
    
def visualize_model_performance(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Model Performance: {type(model).__name__}", fontsize=14)
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring="r2")
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    axes[0, 0].plot(train_sizes, train_mean, 'o-', label="Train Score")
    axes[0, 0].plot(train_sizes, test_mean, 'o-', label="Test Score")
    axes[0, 0].set_title("Learning Curve")
    axes[0, 0].set_xlabel("Training Samples")
    axes[0, 0].set_ylabel("R² Score")
    axes[0, 0].legend()
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, color="blue")
    axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--r")
    axes[0, 1].set_title("True vs Predicted (Test)")
    axes[0, 1].set_xlabel("True Values")
    axes[0, 1].set_ylabel("Predicted Values")
    axes[0, 2].scatter(y_pred_test, residuals_test, alpha=0.5, color="purple")
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title("Residuals vs Predicted (Test)")
    axes[0, 2].set_xlabel("Predicted Values")
    axes[0, 2].set_ylabel("Residuals")
    sns.histplot(residuals_test, bins=30, kde=True, ax=axes[1, 0], color="teal")
    axes[1, 0].set_title("Test Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals")
    sns.histplot(residuals_train, bins=30, kde=True, ax=axes[1, 1], color="green", alpha=0.7)
    axes[1, 1].set_title("Train Residuals Distribution")
    axes[1, 1].set_xlabel("Residuals")
    stats.probplot(residuals_test, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title("QQ Plot (Test Residuals)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def hyperparameter_tuning(models, param_grids, X_train, y_train, scoring_metric='neg_mean_squared_error', cv_folds=5):
    best_models = {}
    best_params = {}
    execution_times = {}
    for model_name, model in models.items():
        print(f"Starting grid search for {model_name}...")
        start_time = time.time()
        if model_name in param_grids:
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param_grids[model_name],
                                       scoring=scoring_metric,
                                       cv=cv_folds,
                                       verbose=1,
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            best_params[model_name] = grid_search.best_params_
            execution_times[model_name] = time.time() - start_time
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Completed grid search for {model_name} in {execution_times[model_name]:.2f} seconds.\n")
        else:
            print(f"No parameter grid available for {model_name}.")
    return best_models, best_params, execution_times

X_train_ols = sm.add_constant(X_train_final)
X_test_ols = sm.add_constant(X_test_final)
ols_model = sm.OLS(y_train, X_train_ols).fit()
print(ols_model.summary())

y_pred = ols_model.predict(X_test_ols)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print("\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

base_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = []
for model_name, model in base_models.items():
    result = evaluate_regression_model(model, X_train_final, y_train, X_test_final, y_test)
    results.append(result)
base_results = pd.DataFrame(results)

base_results

param_grids = {
    "Lasso": {
        "alpha": np.logspace(-4, 1, 10)
    },
    "ElasticNet": {
        "alpha": np.logspace(-4, 1, 10),
        "l1_ratio": np.linspace(0.1, 1, 10)
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7]
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0]
    }
}
tune_models = {
    "Lasso": Lasso(max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(max_iter=10000, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "XGBRegressor": XGBRegressor(random_state=42)
}

best_models, best_params, execution_times = hyperparameter_tuning(tune_models, param_grids, X_train_final, y_train)
for model_name, params in best_params.items():
    print(f"Best parameters for {model_name}: {params}")

results = []
for model_name, model in best_models.items():
    result = evaluate_regression_model(model, X_train_final, y_train, X_test_final, y_test)
    result["Model Name"] = model_name  
    results.append(result)
tuned_results = pd.DataFrame(results)

tuned_results

base_results

tuned_results

final_model = Lasso(alpha=10.0, max_iter=10000, random_state=42)
final_model.fit(X_train_final, y_train)

y_pred_test = final_model.predict(X_test_final)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print(f"Lasso Final Model (alpha=10.0) - Test RMSE: {rmse_test:.2f}")
print(f"Lasso Final Model (alpha=10.0) - Test R²: {r2_test:.3f}")

from sklearn.model_selection import cross_val_score, KFold
cv_folds = 5
cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train_final, y_train, cv=cv, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores):.4f}, Standard Deviation: {np.std(cv_scores):.4f}")
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores, vert=True, patch_artist=True)
plt.title("Cross-Validation R² Scores for Final Lasso Model")
plt.ylabel("R² Score")
plt.xticks([1], ["Lasso"])
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

coefficients = pd.Series(final_model.coef_, index=X_train_final.columns)
coefficients_sorted = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
print("Feature Importances (by coefficient magnitude):")
print(coefficients_sorted)
plt.figure(figsize=(10, 6))
coefficients_sorted.plot(kind='bar')
plt.title("Feature Importance based on Lasso Coefficients (Sorted by |Coefficient|)")
plt.ylabel("Coefficient Value")
plt.show()

loadings = pca.components_
loading_df = pd.DataFrame(
    loadings, 
    columns=retain_features, 
    index=[f"PCA_{i+1}" for i in range(pca.n_components_)]
)
print("PCA Loadings (each row is a principal component):")
loading_df

fig, axs = plt.subplots(1, loading_df.shape[0], figsize=(4 * loading_df.shape[0], 6))
for i, pc in enumerate(loading_df.index):
    sorted_loadings = loading_df.loc[pc].sort_values(ascending=False)
    axs[i].bar(sorted_loadings.index, sorted_loadings.values, color='skyblue')
    axs[i].set_title(pc)
    axs[i].tick_params(axis='x', rotation=90)
    axs[i].grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

visualize_model_performance(final_model, X_train_final, y_train, X_test_final, y_test)

joblib.dump(final_model, 'final_lasso_model.joblib')
print("Model saved as final_lasso_model.joblib")

loaded_model = joblib.load('final_lasso_model.joblib')
print("Model loaded from final_lasso_model.joblib")
y_pred_test = loaded_model.predict(X_test_final)
print("Sample prediction:", y_pred_test[:5])

import types
import inspect
user_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ == '__main__']
imported_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ != '__main__']
imported_pkgs = [name for name in globals() if isinstance(globals()[name], types.ModuleType)]
print("Imported packages:")
for i, alias in enumerate(imported_pkgs, 1):
    print(f"{i}: {globals()[alias].__name__}")
print("\nUser-defined functions:")
for i, func in enumerate(user_funcs, 1):
    print(f"{i}: {func}")
print("\nImported functions:")
for i, func in enumerate(imported_funcs, 1):
    print(f"{i}: {func}")