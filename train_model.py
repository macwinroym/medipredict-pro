import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")

# Remove unwanted blank columns
train = train.loc[:, ~train.columns.str.contains("^Unnamed")]
test = test.loc[:, ~test.columns.str.contains("^Unnamed")]

# Features and target
X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved successfully")