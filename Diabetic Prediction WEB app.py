# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:30:48 2023

@author: acer
"""

import numpy as np
import pickle
import streamlit as st

#loaded the saving model

loaded_model = pickle.load(open('E:\\ML Deployment\\Diabetes_model.sav','rb'))

#creating a funcion for prediction

def diabetes_prediction(input_data):
    loaded_model = pickle.load(open('E:\\ML Deployment\\Diabetes_model.sav','rb'))


    #changing the data to numpy array

    input_to_arr = np.asarray(input_data)

    #reshape the array

    inp_reshape = input_to_arr.reshape(1,-1)

    prediction = loaded_model.predict(inp_reshape)

    print(prediction)

    if prediction == 0:
        return "Not Diabetic"
    else:
        return "Diabetic"
    

def main():
    
    # giving a title 
    
    st.title("Diabetes Prediction Web App")
    
    # getting the input data come the user
    
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Number of Glucose")
    BloodPressure = st.text_input("Number of BloodPressure")
    SkinThickness = st.text_input("Number of SkinThickness")
    Insulin = st.text_input("Number of Insulin")
    BMI = st.text_input("Number of BMI")
    DiabetesPedigreeFunction = st.text_input("Number of DiabetesPedigreeFunction")
    Age = st.text_input("Number of Age")
    
    #code for prediction
    
    diagnosis= ''
    
    # Creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__== "__main__":
    main() 