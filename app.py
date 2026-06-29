from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------------
# Load trained model if available
# -------------------------------
model = None

if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -------------------------------
# Prediction Form
# -------------------------------
@app.route("/predict")
def predict_page():
    return render_template("index.html")


# -------------------------------
# Prediction
# -------------------------------
@app.route("/result", methods=["POST"])
def predict():

    try:

        income = float(request.form["income"])
        age = float(request.form["age"])
        family = float(request.form["family"])

        # Dummy feature vector
        features = np.array([[income, age, family]])

        # If trained model exists
        if model is not None:

            prediction = model.predict(features)[0]

            if prediction == 1:
                result = "Credit Card Approved"
            else:
                result = "Credit Card Rejected"

        else:

            # Demo prediction
            if income > 300000:
                result = "Credit Card Approved"
            else:
                result = "Credit Card Rejected"

    except Exception:

        result = "Invalid Input"

    return render_template(
        "result.html",
        prediction=result
    )


# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
