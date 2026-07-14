from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/best_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    gender = int(request.form["gender"])
    married = int(request.form["married"])
    dependents = int(request.form["dependents"])
    education = int(request.form["education"])
    self_employed = int(request.form["self_employed"])
    income = int(request.form["income"])
    co_income = int(request.form["co_income"])
    loan_amount = int(request.form["loan_amount"])
    loan_term = int(request.form["loan_term"])
    credit_history = int(request.form["credit_history"])
    property_area = int(request.form["property_area"])


    # Create DataFrame with same feature names used during training
    data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        income,
        co_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]],
    columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ])


    # Prediction
    prediction = model.predict(data)


    # Result
    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"


    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)