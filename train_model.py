import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load new dataset
data = pd.read_csv("data/StudentsPerformance.csv")

# Create overall score
data["overall_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

# Features
X = data[[
    "math score",
    "reading score",
    "writing score"
]]

# Target
y = data["overall_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Evaluate
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Save model
joblib.dump(model, "student_model.pkl")

print("Model saved successfully!")