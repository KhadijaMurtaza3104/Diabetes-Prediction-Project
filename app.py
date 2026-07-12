# from fastapi import FastAPI
# import pickle

# app = FastAPI()

# model = pickle.load(open("diabetes_model.pkl", "rb"))

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import numpy as np

# app = FastAPI()

# model = pickle.load(open("diabetes_model.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

# class Patient(BaseModel):
#     Pregnancies: int
#     Glucose: float
#     BloodPressure: float
#     SkinThickness: float
#     Insulin: float
#     BMI: float
#     DiabetesPedigreeFunction: float
#     Age: int

# @app.get("/")
# def home():
#     return {"message": "Welcome to Diabetes Prediction API"}

# @app.post("/predict")
# def predict(patient: Patient):

#     data = np.array([[
#         patient.Pregnancies,
#         patient.Glucose,
#         patient.BloodPressure,
#         patient.SkinThickness,
#         patient.Insulin,
#         patient.BMI,
#         patient.DiabetesPedigreeFunction,
#         patient.Age
#     ]])

#     data = scaler.transform(data)

#     prediction = model.predict(data)

#     if prediction[0] == 1:
#         return {"prediction": "Diabetic"}

#     return {"prediction": "Not Diabetic"}
import gradio as gr
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
):
    data = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "🩺 Diabetic"
    else:
        return "✅ Not Diabetic"

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes."
)

demo.launch()