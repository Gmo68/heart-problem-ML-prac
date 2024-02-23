import json
import requests

url = 'http://127.0.0.1:8000/heart_problem_prediction/'

input_data_for_model = {
    'rest_bp': 145,
    'chest_pain': 3,
    'thalassemia': 2,
    'age': 69,
    'fasting_bs': 0,
    'max_hr': 150,
    'exercise_angina': 1,
    'gender': 1,
    'st_slope': 1,
    'cholesterol': 225,
    'st_depression': 1.3,
    'rest_ecg': 2,
    'num_vessels': 2,
    }

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)