import pandas as pd
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

print("Connected to PostgreSQL!")


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


# -----------------------------
# Execute SQL and load into Pandas
# -----------------------------

df = pd.read_sql(query, engine)

print("Data loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())