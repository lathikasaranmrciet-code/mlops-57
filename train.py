
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 3, 4, 5, 6, 7, 1, 8
    ],
    "attendance": [
        40, 50, 55, 60, 70, 75, 80, 90,
        45, 55, 60, 65, 75, 80, 85, 95,
        50, 60, 65, 70, 80, 90, 35, 98
    ],
    "internal_marks": [
        10, 15, 18, 20, 25, 30, 35, 40,
        12, 14, 19, 22, 28, 32, 38, 45,
        16, 20, 23, 27, 33, 39, 8, 48
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Fail", "Pass"
    ]
}
df = pd.DataFrame(data)
X = df[["study_hours", "attendance", "internal_marks"]]
y = df["result"]
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained successfully!")
print("Model saved as model.pkl")
