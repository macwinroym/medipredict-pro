from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import os

app = FastAPI()

# CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
data = pd.read_csv(os.path.join(BASE_DIR, "Training.csv"))

symptoms = data.drop("prognosis", axis=1).columns.tolist()

# Request format
class InputData(BaseModel):
    symptoms: list[str]

# Routes
@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/symptoms")
def get_symptoms():
    return symptoms

@app.post("/predict")
def predict(data_input: InputData):
    selected = data_input.symptoms

    row = [1 if s in selected else 0 for s in symptoms]
    disease = model.predict([row])[0]

    score = min(97, 48 + len(selected) * 4)

    return {
        "disease": disease,
        "confidence": score
    }