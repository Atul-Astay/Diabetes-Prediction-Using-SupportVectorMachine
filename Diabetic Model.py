# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np

#loaded the saving model

loaded_model = pickle.load(open('E:\\ML Deployment\\Diabetes_model.sav','rb'))

input_data = (1,130,30,25,158,40.1,2.278,39)

#changing the data to numpy array

input_to_arr = np.asarray(input_data)

#reshape the array

inp_reshape = input_to_arr.reshape(1,-1)

prediction = loaded_model.predict(inp_reshape)

print(prediction)

if prediction == 0:
    print("Not Diabetic")
else:
    print("Diabetic")