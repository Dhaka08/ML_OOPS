import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset Loaded Successfully!\n")

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split into training and testing sets.\n")

# 3. Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Model training completed.\n")

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(model, "model.joblib")

print("\nModel saved as 'model.joblib'")