
from flask import Flask, request, render_template_string
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Run train.py first."
    )

model = joblib.load(MODEL_PATH)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Result Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f0f4f8;
            margin: 0;
            padding: 40px 15px;
        }
        .container {
            background: white;
            max-width: 500px;
            margin: auto;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px #cccccc;
        }
        h1 { color: #1d4ed8; text-align: center; }
        label { display: block; margin-top: 18px; }
        input {
            width: 100%;
            box-sizing: border-box;
            padding: 11px;
            margin-top: 7px;
        }
        button {
            width: 100%;
            margin-top: 25px;
            padding: 12px;
            background: #2563eb;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .result {
            margin-top: 20px;
            padding: 14px;
            background: #dcfce7;
            text-align: center;
            font-weight: bold;
        }
        .version { text-align: center; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Result Predictor</h1>
        <p class="version">Model Version: {{ version }}</p>

        <form method="POST">
            <label>Study Hours (0-24)</label>
            <input type="number" name="study_hours"
                   min="0" max="24" step="0.5" required>

            <label>Attendance Percentage (0-100)</label>
            <input type="number" name="attendance"
                   min="0" max="100" required>

            <label>Internal Marks (0-100)</label>
            <input type="number" name="internal_marks"
                   min="0" max="100" required>

            <button type="submit">Predict Result</button>
        </form>

        {% if prediction %}
        <div class="result">
            Predicted Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            internal_marks = float(request.form["internal_marks"])

            if not (0 <= study_hours <= 24):
                raise ValueError("Study hours must be between 0 and 24.")
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be between 0 and 100.")
            if not (0 <= internal_marks <= 100):
                raise ValueError("Marks must be between 0 and 100.")

            features = [[study_hours, attendance, internal_marks]]
            prediction = model.predict(features)[0]

        except ValueError:
            prediction = "Invalid input. Please enter valid values."

    return render_template_string(
        HTML,
        prediction=prediction,
        version=os.environ.get("MODEL_VERSION", "1.0")
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
