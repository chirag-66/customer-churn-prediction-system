# Customer Churn Prediction System

> An end-to-end Machine Learning application that predicts customer churn using **XGBoost** and provides model explainability with **SHAP**, delivered through an interactive **Streamlit dashboard**.

---

## Project Overview

Customer churn is a major business problem for subscription-based companies. Identifying customers who are likely to leave allows businesses to take proactive retention actions.

This project builds a complete customer churn prediction workflow:

**Raw Data → Data Understanding → Preprocessing → Model Training → Hyperparameter Tuning → Threshold Optimization → Explainability → Streamlit Application**

The system predicts the probability that a customer will churn and provides an interpretable explanation of the prediction.

---

## Key Features

- Interactive customer churn dashboard
- Individual customer churn prediction
- XGBoost classification model
- Optimized decision threshold for higher churn recall
- SHAP-based model explainability
- Model performance metrics
- Customer-level risk classification
- Retention recommendations for high-risk customers
- Reusable preprocessing pipeline
- Saved model and threshold for application inference
- Streamlit caching for efficient application execution

---

## Machine Learning Approach

### Problem Type

**Binary Classification**

Target variable:


 `0` - Customer will stay
 `1` - Customer will churn

### Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer demographic information, account details, services, contract information, and billing information.

Important features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

##  Machine Learning Workflow

### 1. Data Understanding

Initial analysis included:

- Dataset structure
- Data types
- Missing values
- Target distribution
- Numerical feature distributions
- Categorical feature analysis
- Correlation analysis
- Outlier investigation

### 2. Data Preprocessing

The preprocessing workflow includes:

- Removing the customer ID column
- Converting `TotalCharges` to numeric
- Handling missing values
- Encoding categorical variables
- Processing numerical features
- Keeping preprocessing inside the ML pipeline

### 3. Model Development

Several ensemble models were evaluated:

- Gradient Boosting
- AdaBoost
- XGBoost

XGBoost provided the strongest overall ROC-AUC performance among the evaluated models and was selected for further tuning.

### 4. Hyperparameter Tuning

Randomized hyperparameter search was used to improve the XGBoost model.

Important tuned parameters included:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `gamma`

### 5. Decision Threshold Optimization

Because customer churn is an imbalanced classification problem, the default probability threshold of `0.50` was not ideal for the project's retention-focused objective.

The deployed model uses a threshold of approximately:

```text
0.26
```

This prioritizes identifying more potential churners, resulting in substantially higher recall.

---

##  Model Performance

At the selected threshold, the model achieved approximately:

| Metric   Score |
#-----------------------
| Accuracy  |  75.5% |
| Precision |  52.5% |
| Recall    |  81.8% |
| F1 Score  |  63.9% |
| ROC-AUC   |  85.0% |

### Why Recall Matters

For churn prediction, missing a customer who is actually going to churn can be costly.

Therefore, this project prioritizes **recall** so that the business can identify more potentially lost customers and apply retention strategies.

> **Note:** The threshold was selected during model development using evaluation data. For a strict production-grade evaluation, threshold selection should be performed on a validation/OOF set followed by a final untouched test-set evaluation.

---

##  Model Explainability with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand which features influence predictions.

Top features identified during global analysis include:

1. Contract — Month-to-month
2. Tenure
3. Online Security — No
4. Monthly Charges
5. Internet Service — Fiber optic
6. Tech Support — No
7. Payment Method — Electronic check
8. Contract — Two year
9. Paperless Billing — No
10. Total Charges

The Streamlit prediction interface provides an individual **SHAP waterfall explanation** showing why the model produced a particular prediction.

---

##  Application

The Streamlit application contains two main sections.

###  Dashboard

The dashboard provides:

- Total customers
- Churned customers
- Retained customers
- Churn rate
- Churn distribution
- Churn by contract
- Churn by internet service
- Customer tenure analysis
- Model performance
- Top churn-driving features

###  Prediction

Users can enter customer information and receive:

- Churn probability
- Churn/stay prediction
- Risk level
- Retention recommendation
- Individual SHAP explanation

---

##  Project Structure

```text
customer-churn-predictor/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   └── processed/
│
├── models/
│   ├── churn_model.pkl
│   └── threshold.pkl
    |__ best_xgb_model1.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
    |__ 02_data_preparation.ipynb
    |__ test_prediction.ipynb
│   
│
├── src/
│
└── reports/
```

---

##  Tech Stack

### Programming Language

- Python

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Logistic Regession
- DicisionTreeClassifier
- gradientBoost
- XGBoost

### Explainable AI

- SHAP

### Application

- Streamlit

### Model Persistence

- Joblib

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/chirag-66/customer-churn-prediction-system.git
```

```bash
cd customer-churn-predictor
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .my_env
```

Activate it:

```bash
.venv\Scripts\activate
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

The application will open in your browser.

---

## Example Prediction Workflow

1. Open the application.
2. Navigate to **Prediction**.
3. Enter customer information.
4. Click **Predict Churn**.
5. Review the churn probability.
6. Check the risk level.
7. Review the recommended retention action.
8. Inspect the SHAP waterfall explanation.

---
# Customer Churn Prediction System

An end-to-end Machine Learning application that predicts customer churn using XGBoost and provides model explainability using SHAP.

##  Live Demo

 [Open the Live Streamlit App](customer-churn-prediction-system∙my-new-branch∙app.py)
## Business Interpretation

The model can help a company identify customers who may be at higher risk of leaving.

For example, a high-risk customer could be considered for:

- Personalized retention offers
- Contract incentives
- Technical support
- Service upgrades
- Customer engagement campaigns

The model should be treated as a **decision-support tool**, not as the sole basis for customer decisions.

---

## Future Improvements

Planned improvements for making the project more production-ready:

- [ ] Separate configuration from application code
- [ ] Add automated model training pipeline
- [ ] Add experiment tracking with MLflow
- [ ] Add data validation
- [ ] Add unit and integration tests
- [ ] Add model/version tracking
- [ ] Add FastAPI prediction API
- [ ] Dockerize the application
- [ ] Add CI/CD with GitHub Actions
- [ ] Deploy the application to the cloud
- [ ] Add model/data drift monitoring
- [ ] Add automated retraining workflow
- [ ] Improve dashboard visualizations
- [ ] Add batch prediction support

---

##  Learning Outcomes

Through this project, I worked with:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Imbalanced classification
- Ensemble learning
- XGBoost
- Hyperparameter tuning
- Probability threshold optimization
- Classification metrics
- Model explainability
- SHAP
- Model serialization
- Streamlit application development
- ML project organization

---

##  Author

**Chirag Thakur**

B.Tech Computer Science Engineering  
Machine Learning / Data Science Enthusiast

---

## If You Find This Project Useful

Give the repository a on GitHub and feel free to explore, improve, or contribute to the project.


End-to-end machine learning project for predicting customer churn.
