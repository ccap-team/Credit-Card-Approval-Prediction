from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# =====================================================
# Load Trained Model
# =====================================================

with open("Models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("Models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# =====================================================
# Home Page
# =====================================================

@app.route("/")
def home():
    return render_template("home.html")


# =====================================================
# Prediction Page
# =====================================================

@app.route("/predict")
def predict_page():
    return render_template("index.html")


# =====================================================
# Prediction
# =====================================================

@app.route("/result", methods=["POST"])
def predict():

    try:

        data = {}

        # --------------------------------------------
        # Read every feature expected by the model
        # --------------------------------------------

        for feature in feature_names:

            value = request.form.get(feature)

            if value is None or value == "":
                value = 0

            # Encode categorical values
            if feature in label_encoders:

                encoder = label_encoders[feature]

                if value not in encoder.classes_:
                    value = encoder.classes_[0]

                value = encoder.transform([value])[0]

            else:
                value = float(value)

            data[feature] = value

        # --------------------------------------------
        # Create DataFrame in training order
        # --------------------------------------------

        input_df = pd.DataFrame(
            [[data[col] for col in feature_names]],
            columns=feature_names
        )

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        approval_probability = round(probability[1] * 100, 2)

        rejection_probability = round(probability[0] * 100, 2)

        if prediction == 1:
            result = "Approved"
            confidence = approval_probability
        else:
            result = "Rejected"
            confidence = rejection_probability

    except Exception as e:
        result = "Error"
        confidence = 0
        approval_probability = 0
        rejection_probability = 0

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        approval_probability=approval_probability,
        rejection_probability=rejection_probability
    )


# =====================================================
# Run Application
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
