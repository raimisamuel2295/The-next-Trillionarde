# 🚗 Car Price Predictor — Flask Web App

A simple web app that lets you enter car specifications and get a predicted price from your trained machine learning model.

---

## 📁 Project Structure

```
car_price_app/
├── app.py              ← Main Python file (Flask server)
├── my_model.pkl        ← Your trained ML model
├── car.csv             ← The original dataset
├── requirements.txt    ← Python packages needed
└── templates/
    └── index.html      ← The web page
```

---

## 🚀 How to Run (Step-by-Step for Beginners)

### Step 1 — Make sure Python is installed
Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:
```
python --version
```
You should see something like `Python 3.10.x`. If not, download Python from https://python.org

---

### Step 2 — Install the required packages
In your terminal, navigate to this folder:
```
cd path/to/car_price_app
```
Then install dependencies:
```
pip install -r requirements.txt
```
> ⚠️ **Important:** The model was trained with scikit-learn 1.7.2. Make sure that exact version is installed.

---

### Step 3 — Run the app
```
python app.py
```
You should see:
```
=======================================================
  🚗  Car Price Predictor is running!
  Open your browser and go to: http://127.0.0.1:5000
=======================================================
```

---

### Step 4 — Open in your browser
Go to: **http://127.0.0.1:5000**

Fill in the car details and click **"Predict Price"**!

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install flask` |
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn==1.7.2` |
| Port already in use | Change `app.run(port=5001)` in app.py |
| Model version warning | Make sure scikit-learn version matches 1.7.2 |

---

## 🧠 How it Works

1. You fill in a form on the web page
2. The form sends your inputs to the Python server (Flask)
3. Flask feeds the inputs into the trained Random Forest model
4. The model returns a predicted car price
5. The price is displayed on your screen

That's it! No cloud, no subscriptions — runs entirely on your computer.
