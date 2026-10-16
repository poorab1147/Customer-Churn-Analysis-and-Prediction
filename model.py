import pandas as pd
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PostgreSQL CONNECTION
# ============================================================

DB_USER = "postgres"
DB_PASSWORD = "iamblessed"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_churn"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# 2. LOAD DATA FROM POSTGRESQL
# ============================================================

query = """
SELECT
    c.customer_id,
    c.gender,
    c.age,
    c.tenure_months,
    s.contract_type,
    s.monthly_charges,
    st.support_calls,
    c.churn
FROM customers c
LEFT JOIN subscriptions s
    ON c.customer_id = s.customer_id
LEFT JOIN support_tickets st
    ON c.customer_id = st.customer_id;
"""

df = pd.read_sql(query, engine)

print("Dataset shape:", df.shape)


# ============================================================
# 3. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

X = df[
    [
        "gender",
        "age",
        "tenure_months",
        "contract_type",
        "monthly_charges",
        "support_calls"
    ]
]

y = df["churn"]


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "gender",
    "contract_type"
]

numerical_features = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_calls"
]


# ============================================================
# 5. LOGISTIC REGRESSION PREPROCESSOR
# ============================================================

logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ============================================================
# 6. RANDOM FOREST PREPROCESSOR
# ============================================================

rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            logistic_preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")


# -----------------------------
# Logistic Regression predictions
# -----------------------------

logistic_pred = logistic_model.predict(X_test)

logistic_probability = (
    logistic_model.predict_proba(X_test)[:, 1]
)


# -----------------------------
# Logistic Regression evaluation
# -----------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred
)

logistic_recall = recall_score(
    y_test,
    logistic_pred
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probability
)


print("\n========================================")
print("      LOGISTIC REGRESSION RESULTS")
print("========================================")

print(f"Accuracy : {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall   : {logistic_recall:.4f}")
print(f"F1-score : {logistic_f1:.4f}")
print(f"ROC-AUC  : {logistic_roc_auc:.4f}")


# -----------------------------
# Logistic Regression confusion matrix
# -----------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_pred
)

print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            rf_preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed!")


# -----------------------------
# Random Forest predictions
# -----------------------------

rf_pred = random_forest_model.predict(X_test)

rf_probability = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# -----------------------------
# Random Forest evaluation
# -----------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)


print("\n========================================")
print("          RANDOM FOREST RESULTS")
print("========================================")

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-score : {rf_f1:.4f}")
print(f"ROC-AUC  : {rf_roc_auc:.4f}")


# -----------------------------
# Random Forest confusion matrix
# -----------------------------

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)


# ============================================================
# 10. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# ============================================================
# 11. MLFLOW EXPERIMENT TRACKING
# ============================================================

# The MLflow tracking server must be running at this address.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer-Churn-Prediction")

# Track Logistic Regression.
with mlflow.start_run(run_name="Logistic-Regression"):
    mlflow.log_params({
        "model_type": "Logistic Regression",
        "test_size": 0.20,
        "random_state": 42,
        "max_iter": 1000,
        "threshold": 0.50,
    })

    mlflow.log_metrics({
        "accuracy": float(logistic_accuracy),
        "precision": float(logistic_precision),
        "recall": float(logistic_recall),
        "f1_score": float(logistic_f1),
        "roc_auc": float(logistic_roc_auc),
    })

    mlflow.log_text(
        str(logistic_cm),
        "evaluation/logistic_confusion_matrix.txt",
    )

    mlflow.sklearn.log_model(
        sk_model=logistic_model,
        name="logistic_regression_pipeline",
    )

    print("Logistic Regression logged to MLflow.")

# Track Random Forest.
with mlflow.start_run(run_name="Random-Forest"):
    mlflow.log_params({
        "model_type": "Random Forest",
        "test_size": 0.20,
        "random_state": 42,
        "n_estimators": 200,
        "threshold": 0.50,
    })

    mlflow.log_metrics({
        "accuracy": float(rf_accuracy),
        "precision": float(rf_precision),
        "recall": float(rf_recall),
        "f1_score": float(rf_f1),
        "roc_auc": float(rf_roc_auc),
    })

    mlflow.log_text(
        str(rf_cm),
        "evaluation/random_forest_confusion_matrix.txt",
    )

    mlflow.sklearn.log_model(
        sk_model=random_forest_model,
        name="random_forest_pipeline",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
   )

    print("Random Forest logged to MLflow.")

print("Both model runs have been recorded in MLflow.")


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Logistic Regression": [
        logistic_accuracy,
        logistic_precision,
        logistic_recall,
        logistic_f1,
        logistic_roc_auc
    ],
    "Random Forest": [
        rf_accuracy,
        rf_precision,
        rf_recall,
        rf_f1,
        rf_roc_auc
    ]
})

print("\n========================================")
print("             MODEL COMPARISON")
print("========================================")

print(comparison.to_string(index=False))
# ============================================================
# CUSTOMER CHURN RISK PREDICTIONS
# ============================================================

# Get churn probability for every customer
all_churn_probability = logistic_model.predict_proba(X)[:, 1]

# Convert probability into a predicted class using 0.50 threshold
all_predicted_churn = (all_churn_probability >= 0.50).astype(int)

# Create prediction dataframe
predictions = df[
    [
        "customer_id",
        "gender",
        "age",
        "tenure_months",
        "contract_type",
        "monthly_charges",
        "support_calls",
        "churn"
    ]
].copy()

predictions["churn_probability"] = all_churn_probability
predictions["predicted_churn"] = all_predicted_churn

# Create risk categories
predictions["risk_category"] = pd.cut(
    predictions["churn_probability"],
    bins=[-0.01, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

# Sort highest-risk customers first
predictions = predictions.sort_values(
    "churn_probability",
    ascending=False
)

print("\n========================================")
print("       TOP HIGH-RISK CUSTOMERS")
print("========================================")

print(
    predictions[
        [
            "customer_id",
            "contract_type",
            "tenure_months",
            "support_calls",
            "monthly_charges",
            "churn_probability",
            "risk_category"
        ]
    ].head(20).to_string(index=False)
)

print("\n========================================")
print("          RISK DISTRIBUTION")
print("========================================")

print(
    predictions["risk_category"]
    .value_counts()
    .sort_index()
)

# Save predictions for Power BI
# ============================================================
# FINAL POWER BI DATASET
# ============================================================

powerbi_data = predictions[
    [
        "customer_id",
        "gender",
        "age",
        "tenure_months",
        "contract_type",
        "monthly_charges",
        "support_calls",
        "churn",
        "churn_probability",
        "predicted_churn",
        "risk_category"
    ]
].copy()

# Business-friendly fields
powerbi_data["annual_charges"] = (
    powerbi_data["monthly_charges"] * 12
)

powerbi_data["churn_probability_percent"] = (
    powerbi_data["churn_probability"] * 100
)

# Rename target for easier understanding in Power BI
powerbi_data["actual_churn"] = powerbi_data["churn"]

powerbi_data = powerbi_data.drop(
    columns=["churn"]
)

# Save final dashboard dataset
powerbi_data.to_csv(
    "churn_powerbi.csv",
    index=False
)

print("\n========================================")
print("       POWER BI DATASET CREATED")
print("========================================")

print("Rows:", len(powerbi_data))
print("Columns:", len(powerbi_data.columns))

print("\nColumns:")
print(powerbi_data.columns.tolist())

print("\nSaved as: churn_powerbi.csv")

print("\nPrediction file saved as churn_predictions.csv")
# ============================================================
# RISK CATEGORY VALIDATION
# ============================================================

risk_validation = (
    predictions
    .groupby("risk_category", observed=False)
    .agg(
        customer_count=("customer_id", "count"),
        actual_churned=("churn", "sum"),
        actual_churn_rate=("churn", "mean"),
        average_churn_probability=("churn_probability", "mean")
    )
    .reset_index()
)

risk_validation["actual_churn_rate"] = (
    risk_validation["actual_churn_rate"] * 100
)

risk_validation["average_churn_probability"] = (
    risk_validation["average_churn_probability"] * 100
)

print("\n========================================")
print("        RISK CATEGORY VALIDATION")
print("========================================")

print(
    risk_validation.to_string(index=False)
)
