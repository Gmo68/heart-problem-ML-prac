from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import json


app = FastAPI()

class model_input(BaseModel):
    
    rest_bp: int
    chest_pain: int
    thalassemia: int
    age: int
    fasting_bs: int
    max_hr: int
    exercise_angina: int
    gender: int
    st_slope: int
    cholesterol: int
    st_depression: float
    rest_ecg: int
    num_vessels: int

model = load_model('heart_model.h5')

@app.post('/heart_problem_prediction')
def heart_problem_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    rest_bp = input_dictionary['rest_bp']
    chest_pain = input_dictionary['chest_pain']
    thalassemia = input_dictionary['thalassemia']
    age = input_dictionary['age']
    fasting_bs = input_dictionary['fasting_bs']
    max_hr = input_dictionary['max_hr']
    exercise_angina = input_dictionary['exercise_angina']
    gender = input_dictionary['gender']
    st_slope = input_dictionary['st_slope']
    cholesterol = input_dictionary['cholesterol']
    st_depression = input_dictionary['st_depression']
    rest_ecg = input_dictionary['rest_ecg']
    num_vessels = input_dictionary['num_vessels']
    
    
    input_list = [
        rest_bp, 
        chest_pain,
        thalassemia,
        age,
        fasting_bs,
        max_hr,
        exercise_angina,
        gender,
        st_slope,
        cholesterol,
        st_depression,
        rest_ecg,
        num_vessels
    ]
    
    prediction = model.predict([input_list])
    prediction_label = [np.argmax(prediction)]

    if (prediction_label[0] == 0):
        return 'No heart problems detected'
    else:
        return 'Heart problem detected'