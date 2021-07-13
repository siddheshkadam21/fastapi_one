# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:06:12 2021

@author: SIDDHESH
"""

import uvicorn 
from fastapi import FastAPI
from model_names import model_names
import numpy as np
import pickle
import pandas as pd

app= FastAPI()
pickle_in = open("classification.pkl","rb")
classification=pickle.load(pickle_in)

@app.get('/')
def index():
    return{"message": "Hello, boss"}

@app.get('/{name}')
def get_name(name: str):
    return{f'{name}':"You have deployed your model using fast API"}

@app.post("/predict")
def predict_class(data:model_names):
    data=data.dict()
    Age = data["Age"]
    Gender = data["Gender"]
    Education = data["Education"]  
    Live_with = data["Live_with"]
    Motive_about_drug =data["Motive_about_drug"] 
    Spend_most_time= data["Spend_most_time"]  
    Failure_in_life = data["Failure_in_life"]
    Mental_emotional_problem= data["Mental_emotional_problem"]
    Suicidal_thoughts=data["Suicidal_thoughts"]
    Family_relationship=data["Family_relationship"]
    Financials_of_family=data["Financials_of_family"]
    Addicted_person_in_family= data["Addicted_person_in_family"]
    Withdrawal_symptoms=data["Withdrawal_symptoms"]
    Satisfied_with_workplace=data["Satisfied_with_workplace"]
    Case_in_court=data["Case_in_court"]
    Living_with_drug_user=data["Living_with_drug_user"]
    Smoking =data["Smoking"]
    Ever_taken_drug=data["Ever_taken_drug"]
    Friends_influence=data["Friends_influence"]
    If_chance_given_to_taste_drugs=data["If_chance_given_to_taste_drugs"]
    Easy_to_control_use_of_drug=data["Easy_to_control_use_of_drug"]
    prediction=classification.predict([[Age, Gender, Education, Live_with, Motive_about_drug,
       Spend_most_time, Failure_in_life, Mental_emotional_problem,
       Suicidal_thoughts, Family_relationship, Financials_of_family,
       Addicted_person_in_family, Withdrawal_symptoms,
       Satisfied_with_workplace, Case_in_court, Living_with_drug_user,
       Smoking, Ever_taken_drug, Friends_influence,
       If_chance_given_to_taste_drugs, Easy_to_control_use_of_drug]])
    if (prediction==1):
        prediction="one"
    elif(prediction==2):
        prediction="two"
    elif(prediction==3):
        prediction="three"
    else:
        prediction="four"
    return {
        "prediction":prediction
        }
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1',port=8000)
    
# uvicorn app:app --reload