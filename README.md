# Customer Churn Analysis & Prediction

An end-to-end customer churn analysis and machine learning project using PostgreSQL, Python, SQL, and Power BI.

## Project Overview

The goal of this project is to analyze customer churn patterns and build a machine learning model that predicts the probability of customer churn.

The project follows a complete analytics pipeline:

PostgreSQL → SQL → Python → EDA → Feature Engineering → Machine Learning → Churn Probability → Power BI

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PostgreSQL
- SQLAlchemy
- Power BI
- Git & GitHub

## Dataset

The project uses a synthetically generated dataset containing 5,000 customers.

The PostgreSQL database contains three related tables:

- `customers`
- `subscriptions`
- `support_tickets`

The final analytical dataset contains:

- Customer demographics
- Tenure
- Contract type
- Monthly charges
- Support calls
- Actual churn status

## SQL Analysis

SQL was used to:

- Join customer, subscription, and support data
- Calculate overall churn rate
- Analyze churn by support calls
- Aggregate customer-level information
- Prepare the dataset for Python analysis

## Exploratory Data Analysis

The analysis examined churn across:

- Contract type
- Gender
- Tenure
- Support calls
- Monthly charges

Key findings included:

- Month-to-month customers had a churn rate of approximately 55.8%.
- One-year customers had a churn rate of approximately 30.4%.
- Two-year customers had a churn rate of approximately 28.1%.
- Customers with 0–6 months of tenure had a churn rate of approximately 59.1%.
- Churn generally increased as support-call frequency increased.
- Churned customers had higher average monthly charges than customers who stayed.

These findings represent associations in the dataset and should not be interpreted as causal relationships.

## Machine Learning

Two binary classification models were trained and compared:

1. Logistic Regression
2. Random Forest

The dataset was divided into:

- 80% training data
- 20% test data

Stratified splitting was used to preserve the class distribution.

### Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 67.60% | 64.60% |
| Precision | 65.50% | 61.04% |
| Recall | 55.35% | 53.53% |
| F1-score | 60.00% | 57.04% |
| ROC-AUC | 73.13% | 67.81% |

Logistic Regression performed better across all evaluated metrics and was selected as the final model.

## Churn Risk Prediction

The Logistic Regression model was used to generate customer-level churn probabilities.

Customers were grouped into three risk categories:

- Low Risk: probability below 30%
- Medium Risk: probability from 30% to 60%
- High Risk: probability above 60%

The resulting customer risk distribution was:

| Risk Category | Customers |
|---|---:|
| Low | 1,368 |
| Medium | 2,596 |
| High | 1,036 |

The risk categories were also validated against observed churn:

| Risk Category | Actual Churn Rate |
|---|---:|
| Low | 21.05% |
| Medium | 45.49% |
| High | 69.88% |

## Power BI Dashboard

The Power BI dashboard provides:

- Total customer count
- Churned customer count
- Overall churn rate
- High-risk customer count
- Churn rate by contract type
- Churn rate by tenure
- Churn rate by support calls
- Customer churn risk distribution
- High-risk customer table
- Interactive filtering by contract type and gender

## Project Structure

```text
customer-churn-project/
│
├── generate_data.py
├── load_data.py
├── clean_data.py
├── eda.py
├── eda_visuals.py
├── model.py
│
├── churn_predictions.csv
├── churn_powerbi.csv
│
├── Customer_Churn_Dashboard.pbix
├── README.md
└── .gitignore