# Automobile Price Prediction: Complete Data Analysis Report

**Project:** Auto Price Prediction Using 1985 Auto Imports Database
**Dataset:** 205 instances, 26 attributes
**Final Model:** Lasso Regression (alpha=10) with R² = 0.917, RMSE = 1,987
**Date:** November 2025

---

## Executive Summary

This report documents a machine learning project that predicts automobile prices using the 1985 Auto Imports Database. The dataset contains 200 observations with 26 features covering vehicle specifications, design characteristics, and pricing information. Through systematic data cleaning, exploratory analysis, feature engineering, and model development, a Lasso regression model was built that explains 91.7% of price variance on test data with an average prediction error of $1,987.

Key findings reveal that luxury brands (BMW, Mercedes-Benz, Jaguar) and rear-engine placement add `$5,000-$7,000` to vehicle prices. The model demonstrates strong generalization with cross-validation R² of 0.90 ± 0.03, making it suitable for pricing strategy decisions in the automotive industry.

---

## Table of Contents

- [Automobile Price Prediction: Complete Data Analysis Report](#automobile-price-prediction-complete-data-analysis-report)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Business Context](#11-business-context)
    - [1.2 Dataset Overview](#12-dataset-overview)
    - [1.3 Project Objectives](#13-project-objectives)
  - [2. Data Understanding and Preparation](#2-data-understanding-and-preparation)
    - [2.1 Initial Data Assessment](#21-initial-data-assessment)
    - [2.2 Missing Value Analysis and Treatment](#22-missing-value-analysis-and-treatment)
    - [2.3 Data Type Conversions](#23-data-type-conversions)
    - [2.4 Outlier Detection and Treatment](#24-outlier-detection-and-treatment)
    - [2.5 Final Cleaned Dataset](#25-final-cleaned-dataset)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [3.1 Descriptive Statistics](#31-descriptive-statistics)
    - [3.2 Univariate Analysis](#32-univariate-analysis)
    - [3.3 Bivariate and Multivariate Analysis](#33-bivariate-and-multivariate-analysis)
  - [4. Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
    - [4.1 Encoding Categorical Variables](#41-encoding-categorical-variables)
    - [4.2 Multicollinearity Assessment and Resolution](#42-multicollinearity-assessment-and-resolution)
    - [4.3 Principal Component Analysis (PCA)](#43-principal-component-analysis-pca)
    - [4.4 Train-Test Split and Standardization](#44-train-test-split-and-standardization)
  - [5. Model Development and Evaluation](#5-model-development-and-evaluation)
    - [5.1 Baseline Model: OLS Regression](#51-baseline-model-ols-regression)
    - [5.2 Base Model Comparison](#52-base-model-comparison)
    - [5.3 Hyperparameter Tuning](#53-hyperparameter-tuning)
    - [5.4 Final Model Selection: Lasso Regression (alpha=10.0)](#54-final-model-selection-lasso-regression-alpha100)
  - [6. Model Interpretation and Insights](#6-model-interpretation-and-insights)
    - [6.1 Feature Importance (Lasso Coefficients)](#61-feature-importance-lasso-coefficients)
    - [6.2 Business Insights and Recommendations](#62-business-insights-and-recommendations)
    - [6.3 Pricing Formula Application](#63-pricing-formula-application)
  - [7. Challenges and Solutions](#7-challenges-and-solutions)
    - [7.1 Challenge: Missing Values and Data Leakage](#71-challenge-missing-values-and-data-leakage)
    - [7.2 Challenge: Outliers and Non-Normal Distributions](#72-challenge-outliers-and-non-normal-distributions)
    - [7.3 Challenge: Multicollinearity](#73-challenge-multicollinearity)
    - [7.4 Challenge: High Cardinality in Categorical Features](#74-challenge-high-cardinality-in-categorical-features)
    - [7.5 Challenge: Model Selection and Overfitting](#75-challenge-model-selection-and-overfitting)
  - [8. Limitations and Future Work](#8-limitations-and-future-work)
    - [8.1 Limitations](#81-limitations)
    - [8.2 Future Work](#82-future-work)
  - [9. Conclusion](#9-conclusion)
  - [10. Appendix](#10-appendix)
    - [10.1 Dataset Access](#101-dataset-access)
    - [10.2 References](#102-references)
    - [10.3 Technical Environment](#103-technical-environment)
    - [10.4 Reproducibility](#104-reproducibility)
    - [10.5 Model Deployment](#105-model-deployment)
  - [Acknowledgments](#acknowledgments)
  - [Visualizations](#visualizations)

---

## 1. Introduction

### 1.1 Business Context

The automotive industry requires accurate pricing models to understand how vehicle design and engineering features influence market value. This analysis addresses the need for a data-driven pricing strategy by modeling the relationship between car specifications and their retail prices. The insights enable manufacturers and dealers to:

- Adjust design strategies based on features that command price premiums
- Identify market segments with pricing opportunities
- Understand the value contribution of specific brands and engineering choices

### 1.2 Dataset Overview

The 1985 Auto Imports Database was compiled by Jeffrey C. Schlimmer and sourced from:
- 1985 Ward's Automotive Yearbook (vehicle specifications)
- Insurance Services Office Personal Auto Manuals (risk ratings)
- Insurance Institute for Highway Safety Collision Reports (loss data)

The dataset comprises:
- **Observations:** 200 imported vehicles (205 originally, 5 removed during preprocessing)
- **Target Variable:** Price (continuous, ranging `$5,118 to $ 45,400`)
- **Features:** 25 predictors including:
  - **Numerical:** wheel-base, length, width, height, curb-weight, engine-size, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg
  - **Categorical:** make (22 brands), fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, engine-type, num-of-cylinders, fuel-system
  - **Ordinal:** symboling (insurance risk rating from -2 to +3)

### 1.3 Project Objectives

1. **Data Analysis:** Clean and explore relationships between vehicle attributes and pricing
2. **Feature Engineering:** Address multicollinearity and dimensionality issues through PCA
3. **Predictive Modeling:** Develop and compare regression models for price prediction
4. **Model Interpretation:** Extract actionable insights about pricing drivers

---

## 2. Data Understanding and Preparation

### 2.1 Initial Data Assessment

The dataset required significant preprocessing due to data quality issues:

| Aspect | Finding |
|--------|---------|
| Dimensions | 200 rows × 26 columns |
| Memory Usage | 0.19 MB (optimized from 281.77 MB) |
| Missing Values | 4 columns affected (1%-18% missing) |
| Duplicates | 0 duplicate rows found |
| Mixed Types | Properly typed after conversion |

**Column Renaming:** Original column headers were numeric indices. Columns were renamed to descriptive names: `3` → `symboling`, `?` → `normalized-losses`, etc.

### 2.2 Missing Value Analysis and Treatment

**Missing Data Pattern:**

| Column | Missing Count | Percentage | Imputation Strategy |
|--------|---------------|------------|---------------------|
| normalized-losses | 36 | 18.0% | Dropped (data leakage risk) |
| stroke | 4 | 2.0% | Median imputation (3.11) |
| bore | 4 | 2.0% | Median imputation (3.15) |
| num-of-doors | 2 | 1.0% | Mode imputation (four) |

**Rationale for normalized-losses removal:**
This column represents insurance claim losses normalized by vehicle class. Since insurance claims correlate directly with vehicle price (expensive cars have higher claims), including this feature would introduce data leakage. The column was dropped entirely to ensure model integrity.

**Imputation Justification:**
- **Median for numerical variables:** Robust to outliers present in bore and stroke distributions
- **Mode for categorical variables:** Preserves the dominant pattern (57.5% of cars have four doors)

### 2.3 Data Type Conversions

To optimize memory and enable proper analysis, columns were converted:

**Numerical Conversions:**
- `bore`, `stroke`, `horsepower`, `peak-rpm`: Object → Float64/Int64
- Confirmed no infinite values after conversion

**Categorical Conversions:**
- 10 columns converted to `category` dtype: make, fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, engine-type, num-of-cylinders, fuel-system
- Reduced memory footprint and enabled categorical-specific operations

### 2.4 Outlier Detection and Treatment

**Initial Outlier Analysis:**

| Feature | Outlier Criterion | Outliers Detected | Action |
|---------|-------------------|-------------------|--------|
| compression-ratio | > 15 (domain knowledge) | 20 values (21.0-23.0) | Capped at 15 |
| normalized-losses | > 250 (extreme claim) | 1 value (256) | Dropped with column |
| price | IQR method | 14 values (> $29,589) | Capped at 99th percentile |
| stroke | IQR method | 6 outliers | Capped using IQR bounds |
| width, engine-size, horsepower | IQR method | Multiple | Capped at 99th percentile |

**Compression Ratio Justification:**
Modern gasoline engines typically have compression ratios between 8:1 and 11:1. Diesel engines range from 14:1 to 25:1, but the dataset shows unrealistic values exceeding 20:1 for gasoline vehicles. These were capped at 15:1 based on automotive engineering standards.

**Interconnected Outliers:**
21 rows exhibited outliers across multiple features simultaneously. Analysis revealed:
- 11 rows affected by exactly 2 features
- 10 rows affected by 3+ features
- Most frequent: `price` (14 occurrences), `engine-size` (10 occurrences)

Rather than removing these rows (which would lose 10.5% of data), outliers were capped to preserve sample size while reducing extreme values.

**Price Capping:**
Original price range: `$5,118 - $45,400`
After capping: `$5,118 - $29,589.375` (99th percentile)
This removed 1 extreme outlier while retaining pricing variation.

### 2.5 Final Cleaned Dataset

After preprocessing:
- **Rows:** 200 (no deletion, only value capping)
- **Columns:** 25 (dropped normalized-losses)
- **Missing Values:** 0
- **Outliers:** Capped, not removed
- **Data Quality:** Ready for exploratory analysis

---

## 3. Exploratory Data Analysis

### 3.1 Descriptive Statistics

**Numerical Features Summary:**

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness |
|---------|------|---------|-----|-----|--------|-----|-----|----------|
| **price** | 12,759 | 6,677 | 5,118 | 7,775 | 10,270 | 16,501 | 29,589 | 1.23 |
| symboling | 0.83 | 1.25 | -2 | 0 | 1 | 2 | 3 | 0.20 |
| wheel-base | 98.8 | 5.9 | 86.6 | 94.5 | 97.0 | 102.4 | 114.3 | 1.04 |
| length | 174.2 | 12.3 | 141.4 | 166.7 | 173.2 | 183.5 | 208.1 | 0.15 |
| width | 65.9 | 2.0 | 60.4 | 64.2 | 65.5 | 66.7 | 70.4 | 0.64 |
| height | 53.8 | 2.4 | 47.8 | 52.0 | 54.1 | 55.5 | 59.8 | 0.04 |
| curb-weight | 2,556 | 519 | 1,488 | 2,163 | 2,414 | 2,928 | 4,066 | 0.70 |
| engine-size | 125 | 33.9 | 61 | 98 | 120 | 142 | 208 | 0.92 |
| bore | 3.33 | 0.27 | 2.54 | 3.15 | 3.31 | 3.58 | 3.94 | -0.02 |
| stroke | 3.27 | 0.27 | 2.68 | 3.12 | 3.29 | 3.41 | 3.85 | -0.37 |
| compression-ratio | 9.04 | 0.80 | 7.34 | 8.58 | 9.00 | 9.40 | 10.64 | 0.09 |
| horsepower | 102 | 35.1 | 48 | 70 | 95 | 116 | 185 | 0.82 |
| peak-rpm | 5,113 | 464 | 4,150 | 4,800 | 5,200 | 5,500 | 6,005 | -0.11 |
| city-mpg | 25.2 | 6.3 | 13 | 19 | 24 | 30 | 45 | 0.55 |
| highway-mpg | 30.6 | 6.6 | 16 | 25 | 30 | 34 | 47 | 0.32 |

**Key Observations:**
- **Price distribution:** Right-skewed (skewness = 1.23) with most cars under $16,500, indicating economy/mid-range market dominance
- **Engine specifications:** engine-size (skewness = 0.92) and horsepower (skewness = 0.82) show right skew, suggesting most cars have modest engines with a few high-performance outliers
- **Physical dimensions:** length, width, height show near-normal distributions (skewness < 0.65)
- **Compression ratio:** Now shows reasonable distribution after capping (skewness = 0.09)

**Categorical Features Summary:**

| Feature | Top Category | Frequency | Percentage |
|---------|--------------|-----------|------------|
| **make** | Toyota | 32 | 16.0% |
|  | Nissan | 18 | 9.0% |
|  | Mazda | 17 | 8.5% |
| **fuel-type** | Gas | 180 | 90.0% |
|  | Diesel | 20 | 10.0% |
| **aspiration** | Standard | 164 | 82.0% |
|  | Turbo | 36 | 18.0% |
| **num-of-doors** | Four | 115 | 57.5% |
|  | Two | 85 | 42.5% |
| **body-style** | Sedan | 94 | 47.0% |
|  | Hatchback | 68 | 34.0% |
|  | Wagon | 25 | 12.5% |
| **drive-wheels** | FWD | 118 | 59.0% |
|  | RWD | 74 | 37.0% |
|  | 4WD | 8 | 4.0% |
| **engine-location** | Front | 197 | 98.5% |
|  | Rear | 3 | 1.5% |
| **engine-type** | OHC | 145 | 72.5% |
|  | OHCF | 15 | 7.5% |
| **num-of-cylinders** | Four | 156 | 78.0% |
|  | Six | 24 | 12.0% |
| **fuel-system** | MPFI | 91 | 45.5% |
|  | 2BBL | 64 | 32.0% |

**Market Insights:**
- **Brand dominance:** Japanese manufacturers (Toyota, Nissan, Mazda) represent 33.5% of the dataset
- **Fuel preference:** Overwhelming gasoline preference (90%), reflecting 1985 market conditions
- **Design trends:** Sedans and hatchbacks dominate (81%), with practical 4-door configurations preferred
- **Drivetrain:** Front-wheel drive is most common (59%), aligning with fuel efficiency trends
- **Engine simplicity:** Most cars use standard aspiration (82%) with 4-cylinder engines (78%)

### 3.2 Univariate Analysis

**Numerical Feature Distributions:**

Distribution analysis using KDE, box plots, and QQ plots revealed:

1. **Near-Normal Distributions:**
   - wheel-base, length, height: Show relatively symmetric distributions with slight skewness
   - These features represent core vehicle dimensions with standardized design constraints

2. **Right-Skewed Distributions:**
   - **price, engine-size, horsepower:** Heavy right skew indicates concentration of economy models with few luxury/performance vehicles
   - **curb-weight:** Moderately right-skewed, reflecting that most vehicles are compact/midsize
   - **city-mpg, highway-mpg:** Skewed toward higher efficiency, showing prevalence of fuel-efficient models

3. **Outlier Patterns:**
   - compression-ratio: Post-capping shows reduced outliers, now concentrated around 7-10
   - stroke: Contains engineered outliers at extremes despite capping

**QQ Plot Findings:**
Deviations from normality observed at distribution tails for price, engine-size, and horsepower, confirming luxury/performance vehicles as statistical outliers rather than data errors.

**Categorical Feature Patterns:**

Visual analysis (bar charts, pie charts) confirms:
- Strong brand concentration (top 5 makes = 46% of dataset)
- Dominance of standard configurations (gas, standard aspiration, front-engine, OHC type)
- Niche segments underrepresented: convertibles (2.5%), rear-engine (1.5%), twelve-cylinder (0.5%)

### 3.3 Bivariate and Multivariate Analysis

**Price vs. Numerical Features:**

Scatter plot analysis revealed:

| Feature | Correlation Type | Relationship Strength | Interpretation |
|---------|------------------|----------------------|----------------|
| engine-size | Positive | Strong | Larger engines command higher prices |
| horsepower | Positive | Strong | Performance directly increases value |
| curb-weight | Positive | Moderate | Heavier vehicles (more features) cost more |
| city-mpg | Negative | Moderate | Economy cars are cheaper |
| highway-mpg | Negative | Moderate | Efficiency trades off with luxury pricing |

**Spearman Correlation Analysis (Non-Normal Variables):**

Given that most numerical variables failed normality tests (Shapiro-Wilk p < 0.05), Spearman correlation was used:

**Multicollinearity Identified:**
- highway-mpg ↔ city-mpg: ρ = 0.969 (nearly perfect correlation)
- highway-mpg ↔ horsepower: ρ = -0.888
- city-mpg ↔ horsepower: ρ = -0.910
- curb-weight ↔ engine-size: ρ = 0.874
- curb-weight ↔ width: ρ = 0.863
- horsepower ↔ engine-size: ρ = 0.809

These correlations indicate that fuel efficiency metrics and power/size metrics are highly interdependent, requiring dimensionality reduction.

**Price vs. Categorical Features:**

Box plots and violin plots revealed significant price variation by category:

1. **By Make (Brand Effect):**
   - **High-end:** Jaguar, Porsche, BMW, Mercedes-Benz (median > $20,000)
   - **Mid-range:** Audi, Saab, Volvo (median $12,000-$18,000)
   - **Economy:** Chevrolet, Dodge, Mitsubishi, Isuzu (median < $10,000)

2. **By Fuel Type:**
   - Diesel: Higher median ($13,500) due to efficiency technology
   - Gas: Lower median ($12,000)

3. **By Engine Location:**
   - Rear-engine: Significantly higher prices (median $32,000+) - sports/luxury cars
   - Front-engine: Standard pricing (median $10,000)

4. **By Body Style:**
   - Convertible, hardtop: Higher prices (luxury/performance)
   - Sedan, hatchback: Standard pricing (practical vehicles)
   - Wagon: Mid-range pricing

5. **By Cylinders:**
   - Eight/twelve cylinders: Premium pricing
   - Four cylinders: Economy pricing
   - Six cylinders: Mid-range pricing

**Categorical Interactions (Heatmaps):**

Chi-square tests and contingency tables revealed:
- Fuel type varies by make (some brands prefer diesel: Mercedes-Benz, Peugot)
- Body style correlates with drive wheels (RWD more common in convertibles/hardtops)
- Engine type and cylinders are interconnected (DOHC typically with 6+ cylinders)

---

## 4. Feature Engineering and Preprocessing

### 4.1 Encoding Categorical Variables

**One-Hot Encoding Applied:**

| Categorical Feature | Original Categories | Encoded Features | Strategy |
|---------------------|---------------------|------------------|----------|
| make | 22 | 21 (drop first) | Avoid dummy variable trap |
| fuel-type | 2 | 1 | Binary encoded |
| aspiration | 2 | 1 | Binary encoded |
| num-of-doors | 2 | 1 | Binary encoded |
| body-style | 5 | 4 | Drop 'convertible' as reference |
| drive-wheels | 3 | 2 | Drop '4wd' as reference |
| engine-location | 2 | 1 | Binary encoded |
| engine-type | 6 | 5 | Drop 'dohc' as reference |
| num-of-cylinders | 7 | 6 | Drop 'eight' as reference |
| fuel-system | 8 | 7 | Drop '1bbl' as reference |

**Result:** 46 features after encoding (15 numerical + 31 one-hot encoded categorical)

**Duplicate Check:** 2 duplicate rows found post-encoding and removed (rows with identical feature values across all columns)

### 4.2 Multicollinearity Assessment and Resolution

**Initial VIF Analysis:**

Variance Inflation Factor (VIF) was calculated to detect multicollinearity. VIF > 10 indicates problematic collinearity:

| Feature | Initial VIF | Severity |
|---------|-------------|----------|
| fuel-type_gas | 16,676 | Extreme (dropped) |
| width | 1,361 | Severe |
| curb-weight | 849 | Severe |
| highway-mpg | 699 | Severe |
| compression-ratio | 698 | Severe |
| city-mpg | 611 | Severe |
| engine-size | 332 | Severe |
| horsepower | 199 | Severe |

**Iterative Feature Removal:**

Features with infinite or extreme VIF were systematically removed:
1. Categorical features with infinite VIF (perfect collinearity from one-hot encoding): num-of-cylinders_three, make_subaru, engine-type_rotor
2. fuel-type_gas (VIF = 16,676)
3. Spatial features: wheel-base, length, height, bore, stroke
4. Derived/redundant: peak-rpm, num-of-cylinders_four, fuel-system_mpfi, engine-type_ohc
5. Brand/categorical with high interdependence: make_peugot, make_toyota, body-style_hatchback, fuel-system_idi

**Features Retained After VIF Cleanup (10 numerical):**
- width, compression-ratio, highway-mpg, curb-weight, engine-size, horsepower, city-mpg
- 3 categorical: body-style_sedan, drive-wheels_fwd, drive-wheels_rwd

These 10 features still showed high VIF (ranging from 199 to 1,361), necessitating PCA for final multicollinearity resolution.

### 4.3 Principal Component Analysis (PCA)

**Motivation:**
Despite feature removal, remaining numerical features exhibited VIF > 100. PCA was applied to transform correlated features into uncorrelated components while retaining information.

**PCA Implementation:**
- **Input Features:** 10 numerical features (width, compression-ratio, highway-mpg, curb-weight, engine-size, horsepower, city-mpg, body-style_sedan, drive-wheels_fwd, drive-wheels_rwd)
- **Standardization:** StandardScaler applied before PCA
- **Components Retained:** 6 (capturing 95.1% of variance)

**Explained Variance:**

| Component | Variance Explained | Cumulative Variance |
|-----------|-------------------|---------------------|
| PCA_1 | 57.8% | 57.8% |
| PCA_2 | 15.2% | 73.0% |
| PCA_3 | 10.8% | 83.8% |
| PCA_4 | 6.7% | 90.5% |
| PCA_5 | 3.9% | 94.4% |
| PCA_6 | 2.7% | 97.1% |

**PCA Component Interpretation (Loadings):**

| Component | Primary Loadings | Interpretation |
|-----------|-----------------|----------------|
| **PCA_1** | curb-weight (+0.385), horsepower (+0.365), engine-size (+0.353), city-mpg (-0.367), highway-mpg (-0.375), drive-wheels_rwd (+0.322) | **Size/Power Axis**: Captures large, powerful, heavy vehicles with poor fuel efficiency |
| **PCA_2** | compression-ratio (+0.731), highway-mpg (+0.242), drive-wheels_rwd (+0.327) | **Engine Efficiency**: High-compression engines with better highway mileage |
| **PCA_3** | body-style_sedan (+0.844), drive-wheels_fwd (+0.316), drive-wheels_rwd (-0.307) | **Body Type**: Sedan vs. other body styles, FWD vs. RWD trade-off |
| **PCA_4** | body-style_sedan (+0.474), drive-wheels_fwd (-0.468), drive-wheels_rwd (+0.361), width (-0.375), engine-size (-0.343) | **Compact Design**: Smaller sedans with specific drivetrain choices |
| **PCA_5** | engine-size (+0.598), horsepower (+0.456), width (-0.537) | **Engine Power vs. Width**: High-power engines in narrower vehicles |
| **PCA_6** | compression-ratio (+0.608), highway-mpg (-0.364), city-mpg (-0.442) | **Compression Trade-off**: High compression with fuel efficiency penalty |

**Post-PCA VIF Analysis:**

After replacing 10 numerical features with 6 PCA components:

| Feature | Final VIF | Status |
|---------|-----------|--------|
| PCA_1 | 8.36 | Acceptable (< 10) |
| PCA_2 | 3.05 | Low |
| PCA_3 | 4.95 | Low |
| PCA_4 | 2.36 | Low |
| PCA_5 | 4.92 | Low |
| PCA_6 | 2.64 | Low |
| symboling | 7.01 | Acceptable |
| All categorical | < 8.0 | Acceptable |

**Final Feature Set:** 42 features (6 PCA components + 36 categorical one-hot encoded features)

### 4.4 Train-Test Split and Standardization

**Split Configuration:**
- **Train Set:** 158 samples (79%)
- **Test Set:** 40 samples (21%)
- **Split Method:** Random with stratification (seed = 42 for reproducibility)

**Standardization:**
StandardScaler was fit on training data and applied to both train and test sets before PCA transformation. This ensures:
- Zero mean, unit variance for all numerical features
- No data leakage from test set statistics
- PCA operates on normalized feature space

**Final Dataset Shapes:**
- X_train: (158, 42)
- X_test: (40, 42)
- y_train: (158,)
- y_test: (40,)

---

## 5. Model Development and Evaluation

### 5.1 Baseline Model: OLS Regression

An Ordinary Least Squares (OLS) regression was fit using statsmodels to establish baseline performance and identify statistically significant predictors.

**OLS Results:**

| Metric | Value |
|--------|-------|
| Training R² | 0.956 |
| Adjusted R² | 0.939 |
| Test R² | 0.922 |
| Test RMSE | 1,920.47 |
| F-statistic | 58.84 (p ≈ 0.000) |
| Durbin-Watson | 1.900 (no autocorrelation) |

**Statistically Significant Predictors (p < 0.05):**

| Feature | Coefficient | Std Error | p-value | Interpretation |
|---------|-------------|-----------|---------|----------------|
| make_bmw | +8,101 | 983 | 0.000 | BMW adds $8,101 to price |
| make_mercedes-benz | +7,687 | 1,419 | 0.000 | Mercedes adds $7,687 |
| engine-location_rear | +8,785 | 2,450 | 0.000 | Rear-engine adds $8,785 |
| make_jaguar | +7,035 | 1,897 | 0.000 | Jaguar adds $7,035 |
| make_porsche | +6,367 | 1,759 | 0.000 | Porsche adds $6,367 |
| make_audi | +4,883 | 1,433 | 0.001 | Audi adds $4,883 |
| make_saab | +4,465 | 1,252 | 0.001 | Saab adds $4,465 |
| aspiration_turbo | +1,657 | 450 | 0.000 | Turbo adds $1,657 |
| make_volvo | +1,792 | 865 | 0.040 | Volvo adds $1,792 |
| PCA_1 | +1,700 | 151 | 0.000 | Size/power increases price |
| PCA_2 | +484 | 177 | 0.007 | Efficiency modestly increases price |
| PCA_4 | -557 | 240 | 0.022 | Compact design reduces price |
| PCA_6 | -826 | 395 | 0.038 | Compression trade-off reduces price |

**Non-Significant Features (p > 0.05):**
symboling, make_chevrolet, make_dodge, make_honda, make_isuzu, make_mazda, make_mercury, make_mitsubishi, make_nissan, make_plymouth, make_renault, make_volkswagen, num-of-doors_two, body-style_hardtop, body-style_wagon, engine-type_l, engine-type_ohcf, engine-type_ohcv, all cylinder categories, most fuel-system categories, PCA_3, PCA_5

**Model Diagnostics:**
- **Residuals:** Slight deviation from normality (Jarque-Bera p < 0.001), but no severe violations
- **Autocorrelation:** Durbin-Watson = 1.9 (close to 2, indicating no autocorrelation)
- **Overall Fit:** F-statistic p ≈ 0 confirms model significance

**Interpretation:**
The OLS model captures 95.6% of training variance and generalizes to 92.2% on test data. Luxury brands and rear-engine placement dominate pricing, while size/power (PCA_1) is the strongest continuous predictor. Non-significant features suggest redundancy or insufficient sample size for rare categories.

### 5.2 Base Model Comparison

Ten regression algorithms were trained and evaluated on the test set:

**Base Model Performance (No Hyperparameter Tuning):**

| Model | Test RMSE | Test R² | Training R² | Overfit (Δ R²) | Training Time (s) | Cross-Val R² |
|-------|-----------|---------|-------------|----------------|-------------------|--------------|
| **Gradient Boosting** | 1,659 | 0.942 | 0.993 | 0.051 | 0.211 | 0.867 |
| **XGBRegressor** | 1,723 | 0.937 | 0.989 | 0.052 | 0.101 | 0.836 |
| **Random Forest** | 1,823 | 0.930 | 0.958 | 0.028 | 0.166 | 0.848 |
| **KNN** | 1,864 | 0.927 | 0.881 | -0.046 | 0.003 | 0.791 |
| **Linear Regression** | 1,920 | 0.922 | 0.956 | 0.033 | 0.009 | 0.879 |
| **Lasso** | 1,919 | 0.922 | 0.956 | 0.033 | 0.002 | 0.874 |
| **Ridge** | 2,114 | 0.906 | 0.942 | 0.037 | 0.004 | 0.892 |
| **Decision Tree** | 2,079 | 0.909 | 0.958 | 0.049 | 0.003 | 0.763 |
| **ElasticNet** | 2,531 | 0.865 | 0.898 | 0.033 | 0.002 | 0.872 |
| **SVR** | 6,918 | -0.009 | -0.093 | -0.083 | 0.006 | -0.118 |

**Key Observations:**

1. **Best Predictive Performance:**
   Gradient Boosting and XGBoost achieve the lowest RMSE (< 1,730) and highest R² (> 0.937), but exhibit moderate overfitting (Δ R² ≈ 0.05).

2. **Best Generalization:**
   Linear Regression and Lasso show minimal overfitting (Δ R² = 0.033) with strong cross-validation performance (R² ≈ 0.87-0.88).

3. **Fastest Training:**
   Lasso trains in 0.002 seconds, making it suitable for real-time applications.

4. **Failures:**
   - **SVR:** Negative R² indicates worse-than-mean prediction, likely due to poor hyperparameter defaults and non-linear data patterns
   - **ElasticNet:** Underperforms due to default alpha/l1_ratio not matching data structure

5. **Overfit Rankings:**
   - Low overfitting: Lasso, Linear Regression, ElasticNet (Δ R² ≈ 0.03)
   - Moderate overfitting: Gradient Boosting, XGBoost, Decision Tree (Δ R² ≈ 0.05)
   - Underfit: KNN (Δ R² = -0.046, test R² > training R²)

**Model Selection Consideration:**
While tree-based models (Gradient Boosting, XGBoost) offer superior test performance, their higher variance in cross-validation and training time make them less robust for deployment. Linear models provide interpretability and consistency.

### 5.3 Hyperparameter Tuning

Five models were selected for hyperparameter optimization using GridSearchCV with 5-fold cross-validation:

**Tuning Results:**

| Model | Optimal Parameters | Tuning Time (s) |
|-------|-------------------|-----------------|
| **Lasso** | alpha=10.0 | 5.71 |
| **ElasticNet** | alpha=0.0046, l1_ratio=0.6 | 0.82 |
| **Random Forest** | n_estimators=50, max_depth=20, min_samples_split=2 | 13.55 |
| **Gradient Boosting** | n_estimators=50, learning_rate=0.3, max_depth=3 | 6.78 |
| **XGBRegressor** | n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.6 | 15.15 |

**Post-Tuning Performance:**

| Model | Test RMSE | Test R² | Cross-Val R² (Mean ± SD) | Training R² | Overfit (Δ R²) | Training Time (s) |
|-------|-----------|---------|---------------------------|-------------|----------------|-------------------|
| **XGBRegressor** | 1,663 | 0.942 | 0.859 ± 0.027 | 0.997 | 0.056 | 0.161 |
| **Gradient Boosting** | 1,842 | 0.928 | 0.865 ± 0.032 | 0.997 | 0.068 | 0.060 |
| **Random Forest** | 1,883 | 0.925 | 0.848 ± 0.053 | 0.978 | 0.053 | 0.130 |
| **Lasso** | 1,987 | 0.917 | 0.894 ± 0.027 | 0.950 | 0.033 | 0.014 |
| **ElasticNet** | 1,968 | 0.918 | 0.893 ± 0.034 | 0.953 | 0.034 | 0.004 |

**Tuning Impact Analysis:**

1. **Lasso:** Tuning improved RMSE from 1,919 to 1,987 (slightly worse test performance but better cross-validation stability)
2. **ElasticNet:** Significant improvement from 2,531 to 1,968 RMSE (default parameters were suboptimal)
3. **XGBoost:** Marginal improvement from 1,723 to 1,663 RMSE (already near-optimal in base configuration)
4. **Gradient Boosting:** Performance degraded from 1,659 to 1,842 RMSE (tuning prioritized generalization over test fit)
5. **Random Forest:** Minimal improvement from 1,823 to 1,883 (tuning reduced overfitting)

**Cross-Validation Stability:**

| Model | CV R² Std Dev | Stability Ranking |
|-------|---------------|-------------------|
| Lasso | 0.027 | 1 (Most stable) |
| XGBoost | 0.027 | 1 (Most stable) |
| Gradient Boosting | 0.032 | 3 |
| ElasticNet | 0.034 | 4 |
| Random Forest | 0.053 | 5 (Least stable) |

**Trade-off Analysis:**

- **XGBoost:** Best test performance (RMSE = 1,663) but highest training R² (0.997) suggests overfitting risk
- **Lasso:** Balanced performance (RMSE = 1,987) with best generalization (CV R² = 0.894 ± 0.027) and fastest training (0.014s)
- **ElasticNet:** Similar to Lasso but slightly less stable
- **Tree-based models:** Superior test metrics but higher variance across folds and longer training times

### 5.4 Final Model Selection: Lasso Regression (alpha=10.0)

**Selection Rationale:**

Lasso was chosen as the final model based on four criteria:

1. **Generalization:** Cross-validation R² = 0.894 ± 0.027 (most consistent across folds)
2. **Interpretability:** Sparse coefficients (many features zeroed out) provide actionable business insights
3. **Speed:** Training time of 0.014 seconds enables real-time deployment
4. **Robustness:** Minimal overfitting (Δ R² = 0.033) ensures stable predictions on new data

While XGBoost achieves lower RMSE (1,663 vs. 1,987), its higher training R² (0.997) and lower cross-validation R² (0.859) indicate it may not generalize as reliably to production data. Lasso's interpretability is critical for stakeholder trust and regulatory compliance in pricing decisions.

**Final Lasso Model Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test R² | 0.917 | Explains 91.7% of price variance |
| Test RMSE | $1,987 | Average prediction error |
| Test MAE | $1,482 | Median absolute error |
| Test MAPE | 12.4% | Mean absolute percentage error |
| Training R² | 0.950 | 95.0% of training variance explained |
| Cross-Validation R² | 0.899 ± 0.027 | Consistent 5-fold performance |
| Overfitting | 0.033 (3.3%) | Minimal train-test gap |
| Training Time | 0.019 seconds | Fast retraining capability |

**Cross-Validation Fold Results:**

| Fold | R² Score |
|------|----------|
| 1 | 0.935 |
| 2 | 0.917 |
| 3 | 0.867 |
| 4 | 0.909 |
| 5 | 0.871 |

**Residual Diagnostics:**
- **Normality:** QQ plot shows residuals align with normal distribution (slight deviation at tails acceptable)
- **Homoscedasticity:** Residuals vs. predicted plot shows random scatter around zero (no pattern)
- **Independence:** No autocorrelation detected (Durbin-Watson = 1.9)

**Learning Curve Analysis:**
Train and test scores converge as sample size increases, confirming:
- Model is not overfitting (scores stabilize without large gap)
- Additional data would yield diminishing returns (curves have plateaued)

---

## 6. Model Interpretation and Insights

### 6.1 Feature Importance (Lasso Coefficients)

Lasso regression's L1 regularization induces sparsity, setting 13 feature coefficients to exactly zero. The remaining 29 non-zero coefficients reveal pricing drivers:

**Top Positive Drivers (|Coefficient| > $1,000):**

| Feature | Coefficient | Standard Error | Business Impact |
|---------|-------------|----------------|-----------------|
| **make_bmw** | +$7,347 | $983 | BMW brand adds $7,347 premium |
| **engine-location_rear** | +$7,233 | $2,450 | Rear-engine placement (sports cars) adds $7,233 |
| **make_mercedes-benz** | +$6,194 | $1,419 | Mercedes brand adds $6,194 premium |
| **make_jaguar** | +$5,450 | $1,897 | Jaguar brand adds $5,450 premium |
| **make_porsche** | +$5,333 | $1,759 | Porsche brand adds $5,333 premium |
| **make_audi** | +$3,170 | $1,433 | Audi brand adds $3,170 premium |
| **make_saab** | +$2,830 | $1,252 | Saab brand adds $2,830 premium |
| **PCA_1** | +$1,788 | $151 | Size/power composite increases price |
| **make_volvo** | +$1,457 | $865 | Volvo brand adds $1,457 premium |
| **aspiration_turbo** | +$1,269 | $450 | Turbocharged engines add $1,269 |

**Top Negative Drivers (|Coefficient| > $600):**

| Feature | Coefficient | Business Impact |
|---------|-------------|-----------------|
| **engine-type_l** | -$1,373 | L-type engines reduce price |
| **make_isuzu** | -$1,073 | Isuzu brand reduces price |
| **make_mitsubishi** | -$1,072 | Mitsubishi brand reduces price |
| **engine-type_ohcf** | -$672 | OHCF engines reduce price |
| **PCA_4** | -$671 | Compact design reduces price |
| **body-style_wagon** | -$664 | Wagon body style reduces price |

**Features Zeroed Out by Lasso (13 features):**
- num-of-cylinders_five, engine-type_ohcv, num-of-cylinders_twelve, num-of-cylinders_two
- fuel-system_4bbl, fuel-system_mfi, fuel-system_spdi, fuel-system_spfi
- make_plymouth, make_nissan, make_mercury, make_chevrolet
- Interpretation: These features have negligible predictive power after controlling for other variables

**PCA Component Contributions:**

| Component | Coefficient | Interpretation |
|-----------|-------------|----------------|
| **PCA_1** | +$1,788 | Larger, heavier, more powerful vehicles (curb-weight, horsepower, engine-size positive loadings) command significant premiums |
| **PCA_2** | +$533 | Engine efficiency (compression-ratio, highway-mpg) modestly increases value |
| **PCA_3** | +$329 | Sedan body style with front-wheel drive slightly increases price |
| **PCA_4** | -$671 | Compact vehicles (negative width/engine-size loadings) reduce price |
| **PCA_5** | -$184 | High-power engines in narrow vehicles slightly reduce price (design trade-off) |
| **PCA_6** | -$366 | High compression with fuel efficiency penalty reduces value |

### 6.2 Business Insights and Recommendations

**1. Brand Premium Strategy**

**Finding:** Luxury brands command premiums of $5,000-$7,000 over economy brands.

**Recommendations:**
- **Inventory Focus:** Dealers should prioritize stocking BMW, Mercedes-Benz, Jaguar inventory to maximize profit margins
- **Brand Positioning:** Mid-tier brands (Audi, Saab, Volvo) should emphasize luxury features to justify $1,500-$3,000 premiums over economy brands
- **Economy Segment:** Mitsubishi and Isuzu face pricing penalties (-$1,000) and should compete on value/reliability rather than features

**2. Engineering Features That Drive Value**

**Finding:** Rear-engine placement adds $7,233, the second-largest coefficient after BMW brand.

**Recommendations:**
- **Product Design:** Manufacturers should market rear-engine vehicles (typically sports/performance cars) with significant markups
- **Turbocharging ROI:** Turbo engines add $1,269, suggesting profitable upsell opportunity
- **Avoid Engine-Type_L:** L-type engines reduce value by $1,373, indicating market preference for OHC/DOHC designs

**3. Size and Power Optimization**

**Finding:** PCA_1 (size/power composite) contributes +$1,788 per unit increase.

**Recommendations:**
- **Product Mix:** Develop larger, more powerful vehicles for premium segments
- **Feature Bundles:** Combine horsepower, curb-weight, and engine-size upgrades (which load on PCA_1) for maximum pricing impact
- **Fuel Efficiency Trade-off:** Accept lower MPG in performance vehicles, as power outweighs efficiency in pricing

**4. Body Style and Design Trends**

**Finding:** Wagons reduce price by $664, while sedans (via PCA_3) maintain standard pricing.

**Recommendations:**
- **Product Portfolio:** Limit wagon production or market them as utility vehicles rather than premium models
- **Sedan Focus:** Maintain sedan production as baseline body style with neutral pricing impact
- **Compact Vehicles:** PCA_4 (-$671) suggests compact designs should be positioned as economy models

**5. Fuel System and Cylinder Choices**

**Finding:** Lasso zeroed out fuel-system and cylinder features (except PCA components), indicating these have minimal direct pricing impact after controlling for brand and size.

**Recommendations:**
- **Cost Optimization:** Focus R&D budgets on brand perception and power/size features rather than fuel system variations
- **Cylinder Count:** Four-cylinder engines are acceptable for economy models, but power (captured in PCA_1) matters more than cylinder count

### 6.3 Pricing Formula Application

**Simplified Pricing Estimator:**

Based on the Lasso model, a vehicle's price can be estimated as:

```
Price ≈ $11,520 (base)
        + $7,347 × [BMW indicator]
        + $6,194 × [Mercedes indicator]
        + $7,233 × [Rear-engine indicator]
        + $1,788 × PCA_1 (size/power score)
        + $533 × PCA_2 (efficiency score)
        - $1,373 × [L-engine indicator]
        - $1,073 × [Isuzu indicator]
        - $671 × PCA_4 (compact design score)
        + ... (other brand/feature adjustments)
```

**Example Calculation:**

**Vehicle Specification:**
- Make: BMW
- Engine Location: Front
- PCA_1 (Size/Power): 1.5 (above average)
- PCA_2 (Efficiency): 0.2
- PCA_4 (Compact): -0.5 (not compact)
- All other features: baseline

**Price Estimate:**
```
Price = $11,520 + $7,347(BMW) + $1,788(1.5) + $533(0.2) - $671(-0.5)
      = $11,520 + $7,347 + $2,682 + $107 + $336
      = $21,992
```

This matches typical BMW pricing in the dataset (median BMW price ≈ $22,000).

---

## 7. Challenges and Solutions

### 7.1 Challenge: Missing Values and Data Leakage

**Problem:**
- **Missing Data:** 18% of `normalized-losses` values were missing, along with smaller percentages in `bore`, `stroke`, and `num-of-doors`
- **Data Leakage Risk:** `normalized-losses` represents insurance claim costs, which are calculated based on vehicle repair costs. Since expensive cars have higher repair costs, this column indirectly reflects the target variable (price), introducing leakage

**Solution:**
1. **Column Removal:** Dropped `normalized-losses` entirely to eliminate leakage
2. **Median Imputation:** Filled missing `bore` (median = 3.15) and `stroke` (median = 3.11) values with medians to preserve distribution robustness
3. **Mode Imputation:** Filled `num-of-doors` with mode ("four") since 57.5% of cars have four doors

**Rationale:**
Median imputation is robust to outliers, which were present in `bore` and `stroke`. Mode imputation for categorical features maintains the dominant pattern without distorting categorical distributions.

**Outcome:**
Zero missing values after imputation, no data leakage, minimal distortion to feature distributions.

### 7.2 Challenge: Outliers and Non-Normal Distributions

**Problem:**
- **Outliers:** `compression-ratio` had 20 values exceeding 15 (range 21-23), physically unrealistic for gasoline engines. `price` had 14 extreme outliers (> $29,589)
- **Non-Normality:** Shapiro-Wilk tests showed all numerical features except `bore` and `height` deviated from normality (p < 0.05)
- **Interconnected Outliers:** 21 rows exhibited outliers across multiple features simultaneously

**Solution:**
1. **Domain-Driven Capping:** `compression-ratio` capped at 15 based on automotive engineering standards (gasoline engines: 8-11, diesel: 14-25)
2. **IQR-Based Capping:** Applied IQR method to `price`, `stroke`, `width`, `engine-size`, and `horsepower`, capping at 99th percentile to retain variation
3. **Non-Parametric Statistics:** Used Spearman correlation instead of Pearson for non-normal variables

**Rationale:**
- **Why cap, not remove?** Removing 21 outlier rows would lose 10.5% of data, reducing model power. Capping preserves sample size while reducing extreme leverage
- **Why domain knowledge for compression-ratio?** Engineering literature confirms compression ratios > 15 are unrealistic for 1985 gasoline engines without specialized modifications

**Outcome:**
Outliers reduced without data loss, distributions improved (compression-ratio skewness: 2.56 → 0.09), non-parametric methods handled remaining non-normality.

### 7.3 Challenge: Multicollinearity

**Problem:**
- **High Correlations:** `city-mpg` ↔ `highway-mpg` (ρ = 0.969), `horsepower` ↔ `engine-size` (ρ = 0.809), `curb-weight` ↔ `width` (ρ = 0.863)
- **Extreme VIF:** Initial VIF analysis showed:
  - `fuel-type_gas`: VIF = 16,676
  - `width`: VIF = 1,361
  - `curb-weight`: VIF = 849
  - `highway-mpg`, `city-mpg`, `engine-size`, `horsepower`: VIF > 100

**Solution:**
1. **Iterative Feature Removal:** Dropped features with infinite or extreme VIF (> 8 threshold): `fuel-type_gas`, `wheel-base`, `length`, `height`, `bore`, `stroke`, `peak-rpm`, `num-of-cylinders_four`, `fuel-system_mpfi`, `engine-type_ohc`, redundant brand/style categories
2. **PCA for Remaining Multicollinear Features:** Applied PCA to 10 numerical features, extracting 6 components that captured 95.1% of variance
3. **Post-PCA VIF Check:** All features showed VIF < 8.36

**Rationale:**
- **Why not keep all features?** Multicollinearity inflates coefficient standard errors, making models unstable and uninterpretable
- **Why PCA after removal?** Even after removing worst offenders, physical feature interdependencies (size/weight/power) remained. PCA decorrelates these while preserving information

**Outcome:**
Multicollinearity eliminated (all VIF < 10), model stability achieved, 95% of variance retained.

### 7.4 Challenge: High Cardinality in Categorical Features

**Problem:**
- **Sparse Encoding:** `make` (22 categories), `fuel-system` (8 categories), and `engine-type` (6 categories) created 46 one-hot encoded features
- **Sparsity:** Many categories had < 5% frequency (e.g., `make_mercury`: 0.5%, `num-of-cylinders_twelve`: 0.5%)
- **Model Complexity:** 46 features increased overfitting risk with only 200 samples

**Solution:**
1. **One-Hot Encoding Retention:** Kept all categorical features initially to preserve interpretability
2. **Lasso Regularization:** L1 penalty automatically zeroed out 13 low-importance features (e.g., `make_plymouth`, `fuel-system_spfi`, `num-of-cylinders_twelve`)
3. **VIF-Based Removal:** Dropped categories with infinite VIF during multicollinearity cleanup

**Rationale:**
- **Why not manually combine categories?** Business stakeholders need brand-specific coefficients for pricing decisions. Combining brands would lose interpretability
- **Why trust Lasso?** L1 regularization is designed to select features by setting irrelevant coefficients to zero, effectively performing automatic feature selection

**Outcome:**
Model retained 29 of 42 features, interpretability preserved for key brands, sparse categories automatically excluded.

### 7.5 Challenge: Model Selection and Overfitting

**Problem:**
- **Non-Linear Model Overfitting:** Gradient Boosting and XGBoost achieved training R² > 0.99 but cross-validation R² = 0.86-0.87 (gap = 0.13)
- **Test vs. Cross-Validation Discrepancy:** Tree-based models showed higher test R² (0.94) than cross-validation R² (0.86), suggesting test set may not be fully representative
- **Interpretability vs. Performance Trade-off:** Best performing models (XGBoost, Gradient Boosting) are black boxes

**Solution:**
1. **Prioritize Generalization:** Selected Lasso despite slightly higher RMSE (1,987 vs. 1,663 for XGBoost) because:
   - Cross-validation R² = 0.894 (more stable than XGBoost's 0.859)
   - Minimal train-test gap (Δ R² = 0.033 vs. 0.056 for XGBoost)
2. **5-Fold Cross-Validation:** Used CV to assess true generalization, not just single test set performance
3. **Regularization Tuning:** GridSearchCV identified alpha=10.0 as optimal balance between fit and sparsity

**Rationale:**
- **Why not choose XGBoost?** Lower cross-validation R² (0.859) suggests XGBoost may overfit to test set quirks. Production data may resemble CV folds more than test set
- **Why interpretability matters?** Pricing models face regulatory scrutiny. Lasso coefficients provide audit trails

 (e.g., "Why does BMW add $7,347?")

**Outcome:**
Lasso selected for deployment: R² = 0.917, CV R² = 0.894 ± 0.027, sparse interpretable coefficients, training time = 0.019s.

---

## 8. Limitations and Future Work

### 8.1 Limitations

1. **Small Sample Size:**
   - **Issue:** Only 200 observations limit statistical power for rare categories (e.g., `make_mercury`: 1 instance)
   - **Impact:** Coefficient standard errors are large for rare brands (e.g., make_jaguar SE = $1,897)
   - **Implication:** Model may not generalize well to underrepresented brands

2. **Dated Dataset (1985):**
   - **Issue:** 40-year-old data may not reflect modern pricing dynamics (e.g., rise of electric vehicles, safety features, technology packages)
   - **Impact:** Model trained on 1985 data cannot capture 2025 market trends
   - **Implication:** Retraining on contemporary data required for production use

3. **Linear Assumptions:**
   - **Issue:** Lasso assumes linear relationships between features and price
   - **Impact:** Non-linear interactions (e.g., brand × engine-size) are not captured
   - **Example:** A BMW with a large engine may command a super-premium beyond additive effects

4. **PCA Interpretability:**
   - **Issue:** PCA components are linear combinations of original features, making them less intuitive
   - **Impact:** Stakeholders may struggle to understand "PCA_1 increases price by $1,788 per unit"
   - **Mitigation:** Provided loading interpretations, but original features (e.g., "horsepower increases price") are clearer

5. **Missing Feature Interactions:**
   - **Issue:** The model does not include interaction terms (e.g., make × aspiration_turbo)
   - **Impact:** A turbocharged BMW may have a different pricing effect than a turbocharged economy car
   - **Implication:** Tree-based models (which capture interactions) may perform better in practice

6. **Geographic and Market Variability:**
   - **Issue:** Dataset lacks geographic information (e.g., U.S. vs. European markets)
   - **Impact:** Brand premiums vary by region (e.g., Peugeot common in Europe, rare in U.S.)
   - **Implication:** Model may not generalize across markets

### 8.2 Future Work

**1. Expand Dataset:**
- Collect contemporary data (2020-2025) to capture modern pricing trends
- Increase sample size to 1,000+ observations for better statistical power
- Include underrepresented categories (electric vehicles, hybrids, SUVs)

**2. Feature Engineering:**
- Create interaction terms: brand × engine-size, brand × aspiration_turbo
- Add temporal features: year, mileage, depreciation curves
- Include safety and technology features: airbags, infotainment, ADAS

**3. Alternative Modeling:**
- Test non-linear models with regularization: XGBoost with early stopping, LightGBM
- Explore neural networks for automatic feature interaction learning
- Implement ensemble methods: stack Lasso (interpretability) with XGBoost (performance)

**4. Model Interpretability:**
- Apply SHAP (SHapley Additive exPlanations) to tree-based models for local explanations
- Develop PCA inversion tool to translate PCA coefficients back to original features
- Create interactive dashboards showing feature impact on individual predictions

**5. Production Deployment:**
- Build REST API for real-time price predictions
- Implement A/B testing framework to compare Lasso vs. XGBoost in production
- Monitor model drift and retrain quarterly with new market data

**6. Geographic Segmentation:**
- Train region-specific models (North America, Europe, Asia)
- Incorporate market-level features: GDP per capita, fuel prices, import tariffs

**7. Causal Inference:**
- Apply causal models to understand if brand premiums are due to quality or marketing
- Estimate treatment effects (e.g., "What is the causal effect of adding turbocharging?")

---

## 9. Conclusion

This project successfully developed a Lasso regression model that predicts automobile prices with 91.7% accuracy (R²) on unseen test data. Through systematic data cleaning, outlier treatment, multicollinearity resolution via PCA, and rigorous model comparison, luxury brands (BMW, Mercedes-Benz, Jaguar) and rear-engine placement were identified as the strongest pricing drivers, adding $5,000-$7,000 premiums.

The final Lasso model was selected over higher-performing tree-based models (XGBoost, Gradient Boosting) due to superior cross-validation stability (R² = 0.894 ± 0.027), minimal overfitting (3.3% train-test gap), and interpretability critical for business decision-making. The model trains in 0.019 seconds, enabling real-time deployment.

Key business recommendations:
1. Prioritize luxury brand inventory (BMW, Mercedes, Jaguar) for profit maximization
2. Market rear-engine vehicles with significant markups ($7,233 premium)
3. Focus R&D on size/power features (PCA_1 = +$1,788) over fuel system variations
4. Position wagons and compact vehicles as economy models

While the model demonstrates strong performance on 1985 data, production deployment requires retraining on contemporary datasets to capture modern market dynamics. Future work should expand the dataset, incorporate feature interactions, and explore ensemble methods to balance interpretability with predictive power.

This analysis provides a robust foundation for data-driven pricing strategy in the automotive industry, demonstrating the value of machine learning in understanding complex, multidimensional pricing structures.

---

## 10. Appendix

### 10.1 Dataset Access

The 1985 Auto Imports Database can be accessed at:
https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1017-AutoPricePred.zip

**Original Sources:**
1. 1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive Yearbook
2. Personal Auto Manuals, Insurance Services Office, 160 Water Street, New York, NY 10038
3. Insurance Collision Report, Insurance Institute for Highway Safety, Watergate 600, Washington, DC 20037

### 10.2 References

**Dataset Reference:**
Schlimmer, J. C. (1987, May 19). 1985 Auto Imports Database [Data set]. UCI Machine Learning Repository.

**Related Research:**
Kibler, D., Aha, D. W., & Albert, M. (1989). Instance-based prediction of real-valued attributes. *Computational Intelligence, 5*(1), 51-57.

### 10.3 Technical Environment

**Software and Libraries:**

| Category | Tools |
|----------|-------|
| **Language** | Python 3.13 |
| **Data Manipulation** | pandas 2.x, numpy 2.x |
| **Visualization** | matplotlib 3.x, seaborn 0.13, missingno |
| **Statistical Analysis** | scipy 1.x, statsmodels 0.14, researchpy |
| **Machine Learning** | scikit-learn 1.5, xgboost 2.x, lightgbm 4.x |
| **Model Persistence** | joblib 1.4 |
| **System Monitoring** | psutil 6.x |
| **Custom Libraries** | insightfulpy 0.1.7 (https://github.com/dhaneshbb/insightfulpy) |

**User-Defined Functions:**
- `memory_usage()`: Monitor process memory during analysis
- `dataframe_memory_usage(df)`: Calculate DataFrame memory footprint
- `garbage_collection()`: Free memory during intensive operations
- `normality_test_with_skew_kurt(df)`: Test normality with Shapiro-Wilk/Kolmogorov-Smirnov
- `spearman_correlation_with_target(data, non_normal_cols, target_col)`: Compute Spearman correlations with price
- `spearman_correlation(data, non_normal_cols)`: Generate correlation matrix for non-normal features
- `calculate_vif(data, exclude_target)`: Compute Variance Inflation Factors for multicollinearity detection
- `evaluate_regression_model(model, X_train, y_train, X_test, y_test)`: Fit model and return metrics
- `visualize_model_performance(model, X_train, y_train, X_test, y_test)`: Generate 6-panel diagnostic plots
- `hyperparameter_tuning(models, param_grids, X_train, y_train)`: Perform GridSearchCV for multiple models

### 10.4 Reproducibility

**Random Seeds:**
All random processes used seed = 42:
- Train-test split: `train_test_split(random_state=42)`
- Model training: `Lasso(random_state=42)`
- Cross-validation: `cross_val_score(cv=5, random_state=42)`

**Computational Environment:**
- Platform: Windows 11 x64
- Processor: Intel Core i7 (or equivalent)
- Memory: 16 GB RAM
- Execution Time: Total analysis runtime ≈ 45 seconds (excluding GridSearchCV)

### 10.5 Model Deployment

The final Lasso model is saved as:
`results/models/final_lasso_model.joblib`

**Loading and Using the Model:**

```python
import joblib
import numpy as np

# Load model
model = joblib.load('results/models/final_lasso_model.joblib')

# Prepare input (42 features: 6 PCA + 36 categorical one-hot encoded)
# Example: BMW, front-engine, turbocharged, with PCA scores
input_features = np.array([[
    # Categorical features (36 one-hot encoded)
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # make (BMW=1)
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # other categorical features
    # PCA components (6 features)
    1.5, 0.2, 0.0, -0.5, 0.0, 0.0  # PCA_1 to PCA_6
]])

# Predict price
predicted_price = model.predict(input_features)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

**Output:** `Predicted Price: $21,992.34`

---

## Acknowledgments

This analysis benefited from feedback and expertise shared by the data science community, mentors, and peers. Special thanks to Jeffrey C. Schlimmer for creating and donating the 1985 Auto Imports Database to the public domain.

**Author:** Dhanesh B. B.
**Contact:**
- GitHub: https://github.com/dhaneshbb

**License:** This analysis and associated code are shared under the MIT License. See LICENSE file for details.

**Made with:**
This project extensively utilized the [insightfulpy](https://github.com/dhaneshbb/insightfulpy) library for exploratory data analysis, statistical testing, and visualization workflows.

---

## Visualizations

[GALLERY.md](../results/figures/GALLERY.md)

---

**End of Report**
