import pandas as pd
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