# Table of Contents

- [Data Analysis](#data-analysis)
    - [Import data](#import-data)
    - [Imports & functions](#imports-functions)
  - [Data understanding and cleaning](#data-understanding-and-cleaning)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Univariate Analysis](#univariate-analysis)
      - [num_analysis](#num-analysis)
      - [cat_analysis](#cat-analysis)
    - [Bivariate & Multivariate Analysis](#bivariate-multivariate-analysis)
- [Predictive Model](#predictive-model)
  - [Preprocessing](#preprocessing)
      - [Encoding](#encoding)
      - [Relation](#relation)
      - [Splitting](#splitting)
  - [Model Development](#model-development)
    - [Model Training & Evaluation](#model-training-evaluation)
    - [Model comparision & Interpretation](#model-comparision-interpretation)
    - [Best Model](#best-model)
    - [Saving the Model](#saving-the-model)
    - [Loading the Model Further use](#loading-the-model-further-use)
- [Table of Contents](#table-of-contents)
- [Acknowledgment](#acknowledgment)
- [Report](#report)
- [Author Information](#author-information)
- [References](#references)
- [Appendix](#appendix)
  - [About data](#about-data)
  - [Source Code and Dependencies](#source-code-and-dependencies)

---

# Acknowledgment  

I would like to express my sincere gratitude to mentors, colleagues, peers, and the data science community for their unwavering support, constructive feedback, and encouragement throughout this project. Their insights, shared expertise, and collaborative spirit have been invaluable in overcoming challenges and refining my approach. I deeply appreciate the time and effort they have dedicated to helping me achieve success in this endeavor

---

# Report

**Data Analysis Report: Automobile Price Prediction**

 1. Introduction
This report outlines the comprehensive analysis performed on the automobile dataset to prepare it for predictive modeling. The dataset comprises 200 observations and 26 features, including both numerical and categorical variables related to vehicle specifications and pricing. The goal was to clean, explore, and transform the data to enhance model performance.

 2. Data Cleaning and Preprocessing

 2.1 Initial Data Overview
- Dataset Dimensions: 200 rows × 26 columns.
- Memory Usage: Reduced from 282.57 MB to 0.19 MB after optimization.
- Column Renaming: Columns were renamed for clarity (e.g., 3 → symboling, ? → normalized-losses).

 2.2 Handling Missing Values
- Missing values (?) in normalized-losses, bore, stroke, horsepower, and peak-rpm were replaced with NaN.
- Imputation:
  - Numerical columns: Median imputation (e.g., horsepower median = 95).
  - Categorical columns: Mode imputation (num-of-doors mode = four).
- Result: Reduced missing values to 0 except for normalized-losses (18% missing), which was later dropped.

 2.3 Data Type Conversion
- Categorical Columns: Converted to category type (e.g., fuel-type, body-style).
- Numerical Columns: bore, stroke, horsepower, and peak-rpm converted to numeric types.



 3. Exploratory Data Analysis (EDA)

 3.1 Descriptive Statistics
- Key Numerical Features:
  - price: Mean = $12,759, Range = $5,118–$29,589.
  - engine-size: Mean = 124.5, Strong positive skew (Skewness = 0.92).
- Categorical Distributions:
  - make: Toyota (16%), Nissan (9%), Mazda (8.5%) dominated.
  - fuel-type: 90% gas-powered vehicles.

 3.2 Outlier Detection and Treatment
- Outliers Identified: In compression-ratio, stroke, and price using IQR.
- Capping: Applied IQR-based capping to:
  
  compression-ratio ≤ 15, stroke ∈ [2.0, 4.0], price ≤ $29,589.
  
- Interconnected Outliers: 21 rows with outliers across multiple features (e.g., high price and engine-size) were adjusted.

 3.3 Correlation Analysis
- Strong Correlations:
  - city-mpg vs. highway-mpg (ρ = 0.97).
  - horsepower vs. engine-size (ρ = 0.81).
- Action: Addressed multicollinearity using VIF and PCA (see Section 4).



 4. Feature Engineering

 4.1 Dimensionality Reduction (PCA)
- Retained Features: width, compression-ratio, highway-mpg, curb-weight, engine-size, horsepower, city-mpg, body-style_sedan, drive-wheels_fwd, drive-wheels_rwd.
- PCA Application:
  - 6 principal components retained (95% variance explained).
  - Explained Variance: 
    
    PCA_1: 57.8%, PCA_2: 15.2%, PCA_3: 10.8%.
    

 4.2 Multicollinearity Mitigation
- VIF Analysis:
  - Initial high-VIF features: width (VIF = 1,361), curb-weight (VIF = 849).
  - Post-PCA: Maximum VIF reduced to 8.36 (PCA_1), within acceptable limits.



 5. Data Splitting and Standardization
- Train-Test Split: 80-20 split (Train: 158 samples, Test: 40 samples).
- Standardization: Applied StandardScaler to retained features before PCA.



 6. Key Findings and Insights
1. Data Quality: Critical missing values in normalized-losses led to its removal.
2. Outlier Impact: Aggressive capping improved distributions without data loss.
3. Categorical Dominance: make and fuel-system showed high cardinality but were retained for predictive value.
4. Multicollinearity: Addressed via PCA, reducing redundancy while preserving 95% variance.



 7. Conclusion
The dataset is now optimized for modeling:
- Low Multicollinearity: Achieved through PCA and VIF-driven feature removal.
- Clean Distributions: Outliers and missing values addressed.
- Relevant Features: 42 final features, including PCA components.

Next Steps: Proceed with regression models (e.g., Linear Regression, Random Forest) to predict price.

*Note: Visualizations (e.g., heatmaps, boxplots) referenced in this report are saved under extracted_images/.*


----

**Final Model Comparison and Report**


 Final Model Comparison and Report

 1. Model Training & Evaluation (OLS Baseline)

 OLS Regression Summary
- Training Performance:  
  - R² = 0.956, Adj. R² = 0.939  
  - Significant predictors: make_bmw, make_mercedes-benz, engine-location_rear, PCA_1, PCA_2, PCA_4, and PCA_6 (p < 0.05).  
- Test Performance:  
  - RMSE = 1,920.47, R² = 0.922  
- Diagnostics:  
  - Residuals show slight non-normality (Jarque-Bera p < 0.001) but no autocorrelation (Durbin-Watson = 1.9).  
  - High F-statistic (58.84, p ≈ 0) confirms overall significance.  

Interpretation:  
The model captures 95.6% of variance in training data and generalizes well (92.2% on test). Luxury brands (BMW, Mercedes), rear-engine placement, and PCA components (size/power) dominate price predictions. Non-significant features (e.g., symboling, num-of-doors) suggest redundancy or multicollinearity.



 2. Base Model Comparison

 Performance Metrics (Test Set)
| Model                   | RMSE    | R²     | Training Time (s) | Overfit (Δ R²) |  
|-------------------------|---------|--------|-------------------|----------------|  
| Gradient Boosting   | 1,659   | 0.942  | 0.118             | 0.051          |  
| XGBRegressor        | 1,723   | 0.937  | 0.143             | 0.052          |  
| Random Forest       | 1,823   | 0.930  | 0.150             | 0.028          |  
| Lasso               | 1,919   | 0.922  | 0.010             | 0.033          |  
| Linear Regression   | 1,920   | 0.922  | 0.025             | 0.033          |  
| KNN                 | 1,864   | 0.927  | 0.003             | -0.046         |  
| SVR                 | 6,918   | -0.009 | 0.008             | -0.083         |  

 Key Observations:
1. Non-linear Models (Gradient Boosting, XGBoost) achieved the best predictive performance (R² > 0.94, RMSE < 1,700) but showed moderate overfitting.  
2. Linear Models (Lasso, OLS) balanced speed (training <0.03s) and generalizability (Δ R² ≈ 0.03).  
3. SVR and ElasticNet underperformed due to non-linear data patterns and hyperparameter sensitivity.  



 3. Hyperparameter-Tuned Models

 Optimized Parameters:
- Lasso: alpha=10.0  
- XGBRegressor: n_estimators=200, learning_rate=0.1, subsample=0.6  
- Gradient Boosting: learning_rate=0.3, n_estimators=50  

 Post-Tuning Performance:
| Model                   | RMSE    | R²     | Cross-Val R² (Mean ± SD) |  
|-------------------------|---------|--------|---------------------------|  
| XGBRegressor        | 1,663   | 0.942  | 0.859 ± 0.027             |  
| Gradient Boosting   | 1,842   | 0.928  | 0.865 ± 0.032             |  
| Lasso               | 1,987   | 0.917  | 0.894 ± 0.027             |  

 Trade-off Analysis:
- XGBoost achieved the lowest RMSE but exhibited higher variance in cross-validation (SD = 0.027).  
- Lasso showed minimal overfitting (Δ R² = 0.033) and stable cross-validation (SD = 0.027), making it the most robust choice.  



 4. Final Model Selection: Lasso (α=10.0)

 Rationale:
- Generalization: Cross-validation R² = 0.899 ± 0.027 (consistent across folds).  
- Interpretability: Sparse coefficients reveal actionable insights (e.g., BMW adds $7,347 to price).  
- Speed: Trains in <0.02s, suitable for real-time applications.  

 Key Features (|Coefficient| > 1,000):
| Feature                | Coefficient | Interpretation                          |  
|------------------------|-------------|-----------------------------------------|  
| make_bmw             | +7,347      | Luxury brand premium.                   |  
| engine-location_rear | +7,233      | Sports/performance vehicle markup.      |  
| make_mercedes-benz   | +6,194      | High-end brand valuation.               |  
| make_jaguar          | +5,450      | Niche luxury segment pricing.           |  
| PCA_1                | +1,788      | Size/power composite (engine size, curb weight). |  

 Diagnostics:
- Residuals: Normally distributed (QQ-plot alignment).  
- Error Distribution: MAE = $1,482 (12.4% MAPE).  
- Visual Checks:  
  - Learning curves confirm stability.  
  - Residuals vs. Predicted shows homoscedasticity.  



 5. PCA Interpretation

 Component Loadings:
- PCA_1: Captures vehicle size/power (curb-weight, engine-size, horsepower).  
- PCA_2: Reflects engine efficiency (compression-ratio, highway-mpg).  
- PCA_4: Contrasts compact design (width, engine-size) vs. sedan body style.  

Business Insight: Larger, powerful vehicles (PCA_1) command price premiums, while fuel efficiency (PCA_2) has mixed market appeal.  



 6. Conclusion & Recommendations

 Model Deployment:
- Lasso (α=10.0) is deployed for its balance of accuracy, speed, and interpretability.  
- File: final_lasso_model.joblib.  

 Strategic Recommendations:
1. Luxury Brands: Prioritize BMW/Mercedes inventory; their coefficients drive significant price premiums.  
2. Engine Placement: Market rear-engine vehicles as high-performance options.  
3. Feature Engineering: Monitor PCA_1 (size/power) trends to align production with demand.  

 Limitations:
- Non-linear interactions (e.g., brand-engine combos) may require tree-based models for finer granularity.  
- Dataset size (n=158) limits complex model training.  



Appendix:  
- Figures: Learning curves, residual plots, PCA loadings (saved externally).  
- Code: Available in Jupyter notebooks (model_training.ipynb, hyperparameter_tuning.ipynb).  



----

**Challenges Faced Report**

 1. Missing Values and Data Leakage  
 Challenge  
- Missing Values: Columns like normalized-losses (18% missing), bore, stroke, and num-of-doors had missing values.  
- Data Leakage: The normalized-losses column, derived from insurance claims tied to car prices, introduced leakage as it indirectly reflected the target variable.  

 Technique & Rationale  
- Imputation:  
  - Categorical columns (e.g., num-of-doors) were filled with the mode.  
  - Numerical columns (e.g., horsepower, peak-rpm) used median imputation to preserve distribution robustness.  
- Column Removal: normalized-losses was dropped entirely to eliminate data leakage.  



 2. Outliers and Non-Normal Distributions  
 Challenge  
- Outliers: Features like compression-ratio (values >15) and price had unrealistic extremes.  
- Non-Normality: Most numerical features (e.g., engine-size, horsepower) exhibited skewness and failed normality tests (Shapiro-Wilk *p* < 0.05).  

 Technique & Rationale  
- Capping:  
  - Domain Knowledge: compression-ratio was capped at 15, aligning with realistic engine specifications.  
  - IQR Method: Outliers in price, width, and others were clipped at the 99th percentile.  
- Non-Parametric Tests: Spearman correlation was used instead of Pearson for non-normal variables.  



 3. Multicollinearity  
 Challenge  
- High Correlations: Features like curb-weight and engine-size (Spearman *ρ* = 0.87) or city-mpg and highway-mpg (*ρ* = 0.97) caused multicollinearity.  
- VIF Scores: Features like width (VIF = 1361) and curb-weight (VIF = 848) showed severe multicollinearity.  

 Technique & Rationale  
- Feature Removal: High-VIF features (e.g., length, wheel-base) were iteratively removed.  
- Dimensionality Reduction: PCA retained 95% variance with 6 components, reducing redundancy while preserving critical information.  



 4. High Cardinality in Categorical Features  
 Challenge  
- Sparse Encoding: Categorical columns like make (22 unique values) and fuel-system (8 categories) led to 46 features after one-hot encoding.  

 Technique & Rationale  
- One-Hot Encoding: Retained interpretability but increased feature space.  
- Regularization: Lasso regression automatically zeroed out less important encoded features (e.g., fuel-system_spfi).  



 5. Model Selection and Overfitting  
 Challenge  
- Overfitting: Non-linear models (e.g., Gradient Boosting) had high training R² (0.99) but lower cross-validation scores (0.86).  

 Technique & Rationale  
- Regularized Linear Models:  
  - Lasso Regression (α=10) was chosen for its sparsity-inducing property, which simplified the model.  
  - Achieved a test R² of 0.917 and RMSE of 1987, with minimal overfitting (training R² = 0.949).  
- Cross-Validation: 5-fold CV confirmed stability (mean R² = 0.899 ± 0.027).  



 Key Outcomes  
| Metric               | Lasso Model Performance |  
|----------------------|-------------------------|  
| Test R²              | 0.917                   |  
| Test RMSE            | 1987                    |  
| Cross-Validation R²  | 0.899 ± 0.027           |  
| Training Time        | 0.019 seconds           |  



 Conclusion  
The project successfully navigated challenges through a combination of domain-driven imputation, outlier capping, PCA for multicollinearity, and Lasso regularization for model robustness. The final model balances interpretability and performance, making it suitable for real-world deployment in pricing strategies.  


---

# Author Information

- Dhanesh B. B.  

- Contact Information:  
    - [Email](dhaneshbb5@gmail.com) 
    - [LinkedIn](https://www.linkedin.com/in/dhanesh-b-b-2a8971225/) 
    - [GitHub](https://github.com/dhaneshbb)

---


# References

**Dataset Reference:**

Schlimmer, J. C. (1987, May 19). *1985 Auto Imports Database* [Data set]. Retrieved from https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1017-AutoPricePred.zip  
*Original sources:*  
1) 1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive Yearbook;  
2) Personal Auto Manuals, Insurance Services Office, 160 Water Street, New York, NY 10038;  
3) Insurance Collision Report, Insurance Institute for Highway Safety, Watergate 600, Washington, DC 20037.

**Related Research Reference:**

Kibler, D., Aha, D. W., & Albert, M. (1989). Instance-based prediction of real-valued attributes. *Computational Intelligence, 5*(1), 51–57.

----

# Appendix

## About data

The analysis in this project is based on the 1985 Auto Imports Database, which was created by Jeffrey C. Schlimmer and donated on May 19, 1987. This dataset provides detailed specifications for imported cars from 1985 and includes critical information on vehicle characteristics, insurance risk ratings, and normalized losses. 

- Entities & Attributes:  
  The dataset captures three types of information:
  - Vehicle Specifications: Such as width, engine size, horsepower, curb weight, and dimensions.
  - Insurance Risk Ratings: Represented by a "symboling" variable that categorizes vehicles based on their risk relative to their price.
  - Normalized Losses: Indicative of the average loss per insured vehicle, normalized within specific car classifications (e.g., two-door, small, station wagon, etc.).

- Attribute Breakdown:  
  The dataset comprises 26 attributes, including:
  - Continuous variables: (e.g., normalized-losses, wheel-base, length, width, engine-size, horsepower, etc.).
  - Nominal variables: (e.g., make, fuel-type, body-style, drive-wheels).
  - Integer attributes: Such as "symboling" and "curb-weight."

- Missing Values:  
  Some attributes in the dataset contain missing values denoted by “?”, which were handled appropriately during the data preprocessing phase.

 Sources and Previous Usage

The 1985 Auto Imports Database was compiled from multiple sources:
1. 1985 Model Import Car and Truck Specifications from the 1985 Ward's Automotive Yearbook.
2. Personal Auto Manuals from the Insurance Services Office, located at 160 Water Street, New York, NY 10038.
3. Insurance Collision Report from the Insurance Institute for Highway Safety at Watergate 600, Washington, DC 20037.

This dataset has been previously used in research, notably by Kibler, Aha, and Albert (1989), who applied instance-based learning techniques for predicting real-valued attributes in automotive pricing (Kibler, Aha, & Albert, 1989).

 Dataset Access

The dataset can be accessed via the following link:  
[1985 Auto Imports Database](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1017-AutoPricePred.zip)


---

## Source Code and Dependencies

In the development of this project, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

---

Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: builtins
2: builtins
3: pandas
4: warnings
5: researchpy
6: matplotlib.pyplot
7: missingno
8: seaborn
9: numpy
10: scipy.stats
11: textwrap
12: logging
13: time
14: statsmodels.api
15: joblib
16: psutil
17: os
18: gc
19: types
20: inspect

User-defined functions:
1: memory_usage
2: dataframe_memory_usage
3: garbage_collection
4: normality_test_with_skew_kurt
5: spearman_correlation_with_target
6: spearman_correlation
7: calculate_vif
8: evaluate_regression_model
9: visualize_model_performance
10: hyperparameter_tuning

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: plot_boxplots
30: kde_batches
31: box_plot_batches
32: qq_plot_batches
33: num_vs_num_scatterplot_pair_batch
34: cat_vs_cat_pair_batch
35: num_vs_cat_box_violin_pair_batch
36: cat_bar_batches
37: cat_pie_chart_batches
38: num_analysis_and_plot
39: cat_analyze_and_plot
40: chi2_contingency
41: fisher_exact
42: pearsonr
43: spearmanr
44: ttest_ind
45: mannwhitneyu
46: linkage
47: dendrogram
48: leaves_list
49: variance_inflation_factor
50: train_test_split
51: cross_val_score
52: learning_curve
53: resample
54: compute_class_weight
55: mean_absolute_error
56: mean_squared_error
57: r2_score
58: mean_absolute_percentage_error
59: mean_squared_log_error
</pre>
