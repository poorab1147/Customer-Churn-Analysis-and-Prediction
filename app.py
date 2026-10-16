import gradio as gr
import mlflow.sklearn
import pandas as pd

# Load the saved scikit-learn pipeline locally; no MLflow server needed.
model = mlflow.sklearn.load_model("churn_model")


def predict_churn(
    gender,
    age,
    tenure_months,
    contract_type,
    monthly_charges,
    support_calls,
):
    if any(value is None for value in [age, tenure_months, monthly_charges, support_calls]):
        return "Please fill in all fields.", "", ""

    customer = pd.DataFrame([{
        "gender": gender,
        "age": int(age),
        "tenure_months": int(tenure_months),
        "contract_type": contract_type,
        "monthly_charges": float(monthly_charges),
        "support_calls": int(support_calls),
    }])

    probability = float(model.predict_proba(customer)[0][1])
    predicted_churn = probability >= 0.50

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return (
        f"{probability:.1%}",
        "Likely to churn" if predicted_churn else "Unlikely to churn",
        risk,
    )


with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("# Customer Churn Analysis & Prediction")
    gr.Markdown(
        "Enter customer details to estimate churn risk "
        "using the trained Logistic Regression model."
    )

    with gr.Row():
        gender = gr.Dropdown(
            choices=["Male", "Female"], label="Gender", value="Male"
        )
        age = gr.Number(label="Age", value=35, precision=0)

    with gr.Row():
        tenure = gr.Number(
            label="Tenure (months)", value=12, precision=0
        )
        contract = gr.Dropdown(
            choices=["Month-to-month", "One year", "Two year"],
            label="Contract type",
            value="Month-to-month",
        )

    with gr.Row():
        charges = gr.Number(label="Monthly charges", value=850.0)
        calls = gr.Number(label="Support calls", value=4, precision=0)

    predict_button = gr.Button("Predict Churn", variant="primary")
    probability_output = gr.Textbox(label="Churn Probability")
    prediction_output = gr.Textbox(label="Prediction")
    risk_output = gr.Textbox(label="Risk Category")

    predict_button.click(
        fn=predict_churn,
        inputs=[gender, age, tenure, contract, charges, calls],
        outputs=[probability_output, prediction_output, risk_output],
    )


if __name__ == "__main__":
    demo.launch()
