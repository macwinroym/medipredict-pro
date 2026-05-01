import pandas as pd
<<<<<<< HEAD
from sklearn.naive_bayes import GaussianNB
import joblib

# Load dataset
df = pd.read_csv("Training.csv")

# 🔥 FIX 1: Remove unwanted columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 🔥 FIX 2: Fill missing values
df = df.fillna(0)

# Split data
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train model
model = GaussianNB()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully!")
=======
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
>>>>>>> 1199a611307889e7b3744e44a46f422f42d27ed7
