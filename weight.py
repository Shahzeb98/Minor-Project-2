# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:53:43 2019

@author: Shahzeb
"""

from keras.models import model_from_json

# load json and create model
json_f = open("model.json", "r")
loaded_model_json = json_f.read()
json_f.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

# Compile model
loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

i = 0
for layer in loaded_model.layers:
    weights = layer.get_weights()
    print('In layer: '+str(i)+'\nWeights are: '+str(weights))