from keras.models import Sequential, Model, load_model, Input, load_model
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
import os
import matplotlib.pyplot as plt

decoder = load_model('models/decoder_2.h5')

other_styles = np.random.rand(0).tolist()

def refresh():
    global other_styles
    other_styles = np.random.rand(0).tolist()

def evaluate(letter_index, arr):
    # generate input
    letter_input = np.zeros((1, 26))
    letter_input[0][letter_index] = 1
    style_input = np.array(arr.tolist() + other_styles).reshape(1, 7)

    pred_y = decoder.predict([letter_input, style_input])

    img = np.clip(pred_y.reshape(40, 24), 0, 1)*255
    img = Image.fromarray(np.uint8(img))
    img.save('out.png')
