import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# -----------------------------
# PostgreSQL connection details
# -----------------------------

DB_USER = "postgres"
DB_PASSWORD = "YOUR_POSTGRES_PASSWORD"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_churn"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Database connection created successfully!")


# -----------------------------
# Generate customer data
# -----------------------------

np.random.seed(42)

N = 5000

# Customer information
customer_id = np.arange(10001, 10001 + N)

gender = np.random.choice(
    ["Male", "Female"],
    size=N
)

age = np.random.randint(
    18, 71,
    size=N
)

tenure_months = np.random.randint(
    1, 73,
    size=N
)

# -----------------------------
# Generate subscription data
# -----------------------------

contract_type = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    size=N,
    p=[0.55, 0.25, 0.20]
)

monthly_charges = np.round(
    np.random.uniform(300, 1800, size=N),
    2
)

# -----------------------------
# Generate support activity
# -----------------------------

support_calls = np.random.poisson(
    lam=2.5,
    size=N
)

support_calls = np.clip(
    support_calls,
    0,
    12
)

# -----------------------------
# Generate churn probability
# -----------------------------

churn_score = (
    -2.0
    + 1.2 * (contract_type == "Month-to-month")
    + 0.8 * (tenure_months <= 6)
    + 0.35 * support_calls
    + 0.0005 * (monthly_charges - 900)
)

churn_probability = 1 / (1 + np.exp(-churn_score))

churn = np.random.binomial(
    1,
    churn_probability
)

# -----------------------------
# Create DataFrames
# -----------------------------

customers_df = pd.DataFrame({
    "customer_id": customer_id,
    "gender": gender,
    "age": age,
    "tenure_months": tenure_months,
    "churn": churn
})

subscriptions_df = pd.DataFrame({
    "subscription_id": np.arange(50001, 50001 + N),
    "customer_id": customer_id,
    "contract_type": contract_type,
    "monthly_charges": monthly_charges
})

support_df = pd.DataFrame({
    "ticket_id": np.arange(90001, 90001 + N),
    "customer_id": customer_id,
    "support_calls": support_calls
})


# -----------------------------
# Insert data into PostgreSQL
# -----------------------------

customers_df.to_sql(
    "customers",
    engine,
    if_exists="append",
    index=False
)

subscriptions_df.to_sql(
    "subscriptions",
    engine,
    if_exists="append",
    index=False
)

support_df.to_sql(
    "support_tickets",
    engine,
    if_exists="append",
    index=False
)

print(f"Successfully inserted {N} customers.")
print("Data generation completed!")