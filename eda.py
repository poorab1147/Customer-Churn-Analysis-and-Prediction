import pandas as pd
from sqlalchemy import create_engine

# -----------------------------
# PostgreSQL connection
# -----------------------------

DB_USER = "postgres"
DB_PASSWORD = "YOUR_POSTGRES_PASSWORD"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_churn"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# Load analytical dataset
# -----------------------------

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

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# -----------------------------
# 1. Overall churn
# -----------------------------

print("\n--- Overall Churn ---")

churn_summary = (
    df["churn"]
    .value_counts()
    .sort_index()
)

print(churn_summary)

print("\nChurn rate:")
print(df["churn"].mean() * 100)


# -----------------------------
# 2. Churn by contract type
# -----------------------------

print("\n--- Churn by Contract Type ---")

contract_churn = (
    df.groupby("contract_type")["churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print(contract_churn)


# -----------------------------
# 3. Churn by gender
# -----------------------------

print("\n--- Churn by Gender ---")

gender_churn = (
    df.groupby("gender")["churn"]
    .mean()
    * 100
)

print(gender_churn)


# -----------------------------
# 4. Churn by tenure group
# -----------------------------

print("\n--- Churn by Tenure Group ---")

df["tenure_group"] = pd.cut(
    df["tenure_months"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=[
        "0-6 months",
        "7-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

tenure_churn = (
    df.groupby("tenure_group", observed=True)["churn"]
    .mean()
    * 100
)

print(tenure_churn)


# -----------------------------
# 5. Churn by support calls
# -----------------------------

print("\n--- Churn by Support Calls ---")

support_churn = (
    df.groupby("support_calls")["churn"]
    .mean()
    * 100
)

print(support_churn)


# -----------------------------
# 6. Monthly charges
# -----------------------------

print("\n--- Monthly Charges by Churn ---")

charges_summary = (
    df.groupby("churn")["monthly_charges"]
    .agg(["mean", "median", "min", "max"])
)

print(charges_summary)