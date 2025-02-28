# Auto Price Prediction: Data Analysis and Predictive Modeling

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Challenges & Solutions](#-challenges--solutions)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

##  Project Overview

**Objective**: Build a predictive model to understand how car prices vary with design and engineering features, enabling strategic adjustments in automotive design and pricing.  

**Dataset**: [1985 Auto Imports Database](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1017-AutoPricePred.zip) with 205 instances and 26 attributes.  

**Key Tasks**:  
1. **Data Analysis**: Clean, explore, and preprocess data.  
2. **Predictive Modeling**: Train and compare regression models.  
3. **Insight Generation**: Translate model results into actionable business strategies.  

---

##  Project Structure

```
├── data
│   ├── 1.1 raw               # Raw datasets (auto_imports.csv)
│   └── 1.2 processed         # Processed training/testing splits
├── docs                      # Problem statement and documentation
├── notebooks                 # Jupyter notebooks for analysis and modeling
├── report                    # Final Report.md and supporting files
├── results
│   ├── 365csv pre-analysis   # EDA outputs (statistics, visualizations)
│   ├── figures               # Saved plots (boxplots, histograms)
│   └── models                # Serialized models (final_lasso_model.joblib)
├── scripts                   # Utility scripts (AutoPrice.py, utility.py)
├── LICENSE
├── README.md
└── requirements.txt          # Python dependencies
```

---

##  Key Features

- **Data Preprocessing**:  
  - Missing value imputation (median/mode).  
  - Outlier capping using IQR and domain knowledge.  
  - PCA for dimensionality reduction (95% variance retained).  

- **Model Development**:  
  - Compared 7 models: **Lasso**, **XGBoost**, **Gradient Boosting**, etc.  
  - Hyperparameter tuning with cross-validation.  

- **Deployment-Ready**:  
  - Best model: **Lasso Regression** (R² = 0.917, RMSE = 1,987).  
  - Interpretable coefficients for business strategy (e.g., BMW adds $7,347 to price).  

---

##  Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/dhaneshbb/AutoPricePred.git
   cd AutoPricePred
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```

---

##  Usage

1. **Data Analysis**:  
   - Run `AutoPricePred_Analysis.ipynb` to explore data cleaning, EDA, and visualizations.  

2. **Model Training**:  
   - The notebook includes code for model comparison, hyperparameter tuning, and saving the best model.  

3. **Inference**:  
   ```python
   import joblib
   model = joblib.load('models/final_lasso_model.joblib')
   prediction = model.predict([input_features])
   ```

---

##  Results

| Model               | Test RMSE | Test R² | Training Time (s) | Overfit (Δ R²) |
|---------------------|-----------|---------|-------------------|----------------|
| **Lasso (α=10)**    | 1,987     | 0.917   | 0.019             | 0.033          |
| XGBoost             | 1,663     | 0.942   | 0.143             | 0.052          |
| Gradient Boosting   | 1,842     | 0.928   | 0.118             | 0.051          |

### Final Model: Lasso Regression (α=10)
- **Rationale**: Balances interpretability, speed, and generalizability.
- **Key Drivers**: Luxury brands (`make_bmw`, `make_mercedes-benz`), engine location, and PCA components.
- **Performance**:
  
| Metric               | Value          |
|----------------------|----------------|
| Test R²              | 0.917          |
| Test RMSE            | 1,987          |
| Cross-Validation R²  | 0.899 ± 0.027  |

**Key Insights**:  
- Luxury brands (BMW, Mercedes) command significant price premiums.  
- Rear-engine vehicles are associated with higher prices.  
- Vehicle size/power (PCA_1) is a critical pricing factor.  

---

##  Challenges & Solutions

| Challenge               | Solution                              |
|-------------------------|---------------------------------------|
| Missing Values (18%) with data Leakage     | Median/mode imputation + column drop. |
| Multicollinearity        | PCA and VIF-based feature removal.    |
| High Cardinality         | Regularization (Lasso) for sparsity.  |
| Model Overfitting        | Cross-validation and hyperparameter tuning. |

---

##  Contributing

Contributions are welcome!  
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/new-feature`.  
3. Commit changes: `git commit -m 'Add new feature'`.  
4. Push to the branch: `git push origin feature/new-feature`.  
5. Submit a pull request.  

---

##  License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details. 

---

**Made with  using [insightfulpy](https://github.com/dhaneshbb/insightfulpy)**
