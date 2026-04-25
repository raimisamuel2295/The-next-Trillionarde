"""
Car Price Predictor - Flask Web App
=====================================
This app lets you fill in car details and predict the price using a trained
machine learning model (Random Forest Regressor).

HOW TO RUN:
  1. Make sure you have Python installed.
  2. Install dependencies:
       pip install flask scikit-learn pandas joblib numpy
  3. Run the app:
       python app.py
  4. Open your browser and go to:
       http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load the trained model ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model.pkl")
model = joblib.load(MODEL_PATH)

# ── Dropdown options (from the original dataset) ───────────────────────────
DROPDOWN_OPTIONS = {
    "fueltype": ["gas", "diesel"],
    "aspiration": ["std", "turbo"],
    "doornumber": ["two", "four"],
    "carbody": ["convertible", "hatchback", "sedan", "wagon", "hardtop"],
    "drivewheel": ["rwd", "fwd", "4wd"],
    "enginelocation": ["front", "rear"],
    "enginetype": ["dohc", "ohcv", "ohc", "l", "rotor", "ohcf", "dohcv"],
    "cylindernumber": ["four", "six", "five", "three", "twelve", "two", "eight"],
    "fuelsystem": ["mpfi", "2bbl", "mfi", "1bbl", "spfi", "4bbl", "idi", "spdi"],
    "CarName": [
        "alfa-romero giulia", "alfa-romero stelvio", "alfa-romero Quadrifoglio",
        "audi 100 ls", "audi 100ls", "audi fox", "audi 5000", "audi 4000",
        "bmw 320i", "bmw x1", "bmw x3", "bmw z4", "bmw x5",
        "chevrolet impala", "chevrolet monte carlo", "chevrolet vega 2300",
        "dodge rampage", "dodge challenger se", "dodge d200",
        "honda civic", "honda civic cvcc", "honda accord cvcc", "honda accord lx",
        "honda civic 1500 gl", "honda accord", "honda prelude",
        "isuzu MU-X", "isuzu D-Max V-Cross",
        "jaguar xj", "jaguar xf", "jaguar xk",
        "mazda rx2 coupe", "mazda rx-4", "mazda 626", "mazda glc", "mazda rx-7 gs",
        "mitsubishi mirage", "mitsubishi lancer", "mitsubishi outlander",
        "nissan gt-r", "nissan rogue", "nissan titan", "nissan leaf",
        "peugeot 504", "peugeot 304", "peugeot 604sl",
        "porsche macan", "porsche cayenne", "porsche boxter",
        "subaru", "subaru dl", "subaru brz",
        "toyota corona", "toyota corolla", "toyota carina", "toyota celica gt",
        "toyota mark ii", "toyota starlet", "toyota cressida",
        "volkswagen rabbit", "volkswagen dasher", "volkswagen super beetle",
        "volvo 144ea", "volvo 244dl", "volvo 264gl",
    ],
}

# ── Default / example values for numeric fields ────────────────────────────
NUMERIC_DEFAULTS = {
    "car_ID": 1,
    "symboling": 0,
    "wheelbase": 98.8,
    "carlength": 168.8,
    "carwidth": 64.1,
    "carheight": 53.3,
    "curbweight": 2548,
    "enginesize": 130,
    "boreratio": 3.47,
    "stroke": 2.68,
    "compressionratio": 9,
    "horsepower": 111,
    "peakrpm": 5000,
    "citympg": 21,
    "highwaympg": 27,
}

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home page — show the prediction form."""
    return render_template(
        "index.html",
        options=DROPDOWN_OPTIONS,
        defaults=NUMERIC_DEFAULTS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, run the model, return the predicted price."""
    try:
        form = request.form

        # Build a single-row DataFrame matching the training feature order
        input_data = {
            # Numeric features
            "car_ID":          float(form.get("car_ID", 1)),
            "symboling":       float(form.get("symboling", 0)),
            "wheelbase":       float(form.get("wheelbase", 98.8)),
            "carlength":       float(form.get("carlength", 168.8)),
            "carwidth":        float(form.get("carwidth", 64.1)),
            "carheight":       float(form.get("carheight", 53.3)),
            "curbweight":      float(form.get("curbweight", 2548)),
            "enginesize":      float(form.get("enginesize", 130)),
            "boreratio":       float(form.get("boreratio", 3.47)),
            "stroke":          float(form.get("stroke", 2.68)),
            "compressionratio":float(form.get("compressionratio", 9)),
            "horsepower":      float(form.get("horsepower", 111)),
            "peakrpm":         float(form.get("peakrpm", 5000)),
            "citympg":         float(form.get("citympg", 21)),
            "highwaympg":      float(form.get("highwaympg", 27)),
            # Categorical features
            "CarName":         form.get("CarName", "toyota corolla"),
            "fueltype":        form.get("fueltype", "gas"),
            "aspiration":      form.get("aspiration", "std"),
            "doornumber":      form.get("doornumber", "four"),
            "carbody":         form.get("carbody", "sedan"),
            "drivewheel":      form.get("drivewheel", "fwd"),
            "enginelocation":  form.get("enginelocation", "front"),
            "enginetype":      form.get("enginetype", "ohc"),
            "cylindernumber":  form.get("cylindernumber", "four"),
            "fuelsystem":      form.get("fuelsystem", "mpfi"),
        }

        df_input = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(df_input)[0]
        predicted_price = round(float(prediction), 2)

        return jsonify({
            "success": True,
            "price": predicted_price,
            "formatted": f"${predicted_price:,.2f}",
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚗  Car Price Predictor is running!")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True)
