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
# SQL query
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

# Load data
df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

# -----------------------------
# Basic inspection
# -----------------------------

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

print("\n--- Statistical summary ---")
print(df.describe())