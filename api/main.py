from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

cwd = os.getcwd()
one_level_up =os.path.dirname(cwd)
model_directory = os.path.join(one_level_up,'models')
models = os.listdir(model_directory)
assert 'xgboost_uncalibrated.joblib' in models
assert 'calibrated_model.joblib' in models


class ReadmissionFeatures(BaseModel):
    race: str
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str = '?'
    diag_3: str = '?'
    number_diagnoses: int
    number_of_diabetes_medications: int
    prior_readmissions: int


app = FastAPI()

calibrated = joblib.load(f'{model_directory}/calibrated_model.joblib')

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(features: ReadmissionFeatures):
    df_columns = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital',
                  'num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient',
                  'diag_1','diag_2','diag_3','number_diagnoses','number_of_diabetes_medications','prior_readmissions',]
    input_data = pd.DataFrame([[features.race,features.gender,features.age,features.admission_type_id,
                            features.discharge_disposition_id,features.admission_source_id,features.time_in_hospital,
                            features.num_lab_procedures,features.num_procedures,features.num_medications,features.number_outpatient,
                            features.number_emergency,features.number_inpatient,features.diag_1,features.diag_2,features.diag_3,
                            features.number_diagnoses,features.number_of_diabetes_medications,features.prior_readmissions,]], columns=df_columns)
    for category in ['race','gender','age','diag_1','diag_2','diag_3']:
        input_data[category] = input_data[category].astype('category')
    
    prediction = calibrated.predict(input_data)
    predicted_probability = calibrated.predict_proba(input_data)
    predict_positive = float(predicted_probability[0][1])
    return {"prediction": float(prediction[0]), 'predicted_probability': predict_positive}