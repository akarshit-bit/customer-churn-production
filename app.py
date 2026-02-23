from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model/churn_model.pkl", "wb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

gender_map = {"Male": 1, "Female": 0}
subscription_map = {"Basic": 0, "Standard":1, "Premium": 2}
contract_map = {"Monthly": 0, "Quarterly": 1, "Yearly": 2}

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        gender = gender_map[request.form["gender"]]
        tenure = float(request.form["tenure"])
        usage = float(request.form["usage"])
        support = float(request.form["support"])
        delay = float(request.form["delay"])
        subscription = subscription_map[request.form["subscription"]]
        contract = contract_map[request.form["contract"]]
        spend = float(request.form["spend"])
        interaction = float(request.form["interaction"])

        features = np.array([[age, gender, tenure, usage, support, delay, subscription, contract, spend, interaction]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        risk_percentage = round(probability * 100, 2)

        if prediction == 1:
            result = "⚠ High Risk of Churn"
        else:
            result = "✅ Low Risk - Customer Likely to Stay"

            return render_template(
                "index.html",
                prediction=result,
                probablity=risk_percentage
            )

    except Exception as e:
        return render_template("index.html", prediction = "Error in Prediction" )

    if __name__ == "__main__":
        app.run()