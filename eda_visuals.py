import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# Load data
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


# -----------------------------
# Churn by contract type
# -----------------------------

contract_churn = (
    df.groupby("contract_type")["churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
contract_churn.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# -----------------------------
# Churn by tenure group
# -----------------------------

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
    .mul(100)
)

plt.figure(figsize=(8, 5))
tenure_churn.plot(kind="bar")
plt.title("Churn Rate by Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# -----------------------------
# Churn by support calls
# -----------------------------

support_churn = (
    df.groupby("support_calls")["churn"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))
support_churn.plot(kind="line", marker="o")
plt.title("Churn Rate by Support Calls")
plt.xlabel("Number of Support Calls")
plt.ylabel("Churn Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Monthly charges by churn
# -----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="churn",
    y="monthly_charges"
)
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()