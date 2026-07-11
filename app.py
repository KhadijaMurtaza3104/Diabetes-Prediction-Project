# from fastapi import FastAPI
# import pickle

# app = FastAPI()

# model = pickle.load(open("diabetes_model.pkl", "rb"))

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Welcome to Diabetes Prediction API"}

@app.post("/predict")
def predict(patient: Patient):

    data = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        return {"prediction": "Diabetic"}

    return {"prediction": "Not Diabetic"}