import streamlit as st
import pandas as pd
import requests
import json

valid_race = pd.read_csv('valid_race.csv')
valid_age = pd.read_csv('valid_age.csv')
valid_gender = pd.read_csv('valid_gender.csv')
valid_diag = pd.read_csv('valid_diag.csv')

st.title('Hospital Readmission Probability')

race = st.selectbox('Race', valid_race['race'].values)
gender = st.selectbox('Gender', valid_gender['gender'].values)
age = st.selectbox('Age', valid_age['age'].values)
adm_type = st.number_input('Admission Type', min_value=None, max_value=None)
disc_type = st.number_input('Discharge Type', min_value=None, max_value=None)
adm_src = st.number_input('Admission Source', min_value=None, max_value=None)
days = st.number_input('Days in Hospital', min_value=None, max_value=None)
labs = st.number_input('Number of Lab Procedures', min_value=None, max_value=None)
total_procs = st.number_input('Number of Total Procedures', min_value=None, max_value=None)
medications = st.number_input('Number of Medications', min_value=None, max_value=None)
outpt = st.number_input('Number of Outpatient Events', min_value=None, max_value=None)
emerg = st.number_input('Number of Emergency Events', min_value=None, max_value=None)
intpt = st.number_input('Number of Inpatient Events', min_value=None, max_value=None)
prim = st.selectbox('Primary Diagnosis',  list(valid_diag['diag'].values) + ['?'])
scnd = st.selectbox('Secondary Diagnosis', list(valid_diag['diag'].values) + ['?'])
trt = st.selectbox('Tertiary Diagnosis', list(valid_diag['diag'].values) + ['?'])
diag = st.number_input('Number of Diagnoses', min_value=None, max_value=None)
diab = st.number_input('Number of Diabetes Medications', min_value=None, max_value=None)
prior = st.number_input('Number of Prior Admissions', min_value=None, max_value=None)

api_input = [race,gender,age,adm_type,disc_type,adm_src,days,
             labs,total_procs,medications,outpt,emerg,intpt,
             prim,scnd,trt,diag,diab,prior]
api_expected_input = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
    'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
    'number_emergency','number_inpatient','diag_1','diag_2','diag_3',
    'number_diagnoses','number_of_diabetes_medications','prior_readmissions',
]
assert len(api_input) == len(api_expected_input)
json_dict = {k:v for k,v in zip(api_expected_input,api_input)}
print(json_dict)

if st.button('Submit Prediction Request'):
    r = requests.post('http://127.0.0.1:8000/predict', json=json_dict)
    predictions = json.loads(r.text)
    st.write("Prediction: ", predictions['prediction'])
    st.write("Predicted Probability: ", predictions['predicted_probability'])