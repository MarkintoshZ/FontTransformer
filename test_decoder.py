from keras.models import Sequential, Model, load_model, Input, load_model
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
import os
import matplotlib.pyplot as plt

model = load_model('autoencoder_1.h5')

# create decoder
encoded = Input(shape=(26,))
style = Input(shape=(16,))

x = model.layers[10]([encoded, style])
for l in model.layers[11:]:
    x = l(x)

decoder = Model([encoded, style], x)

# generate input
letter_input = np.zeros((1, 26))
letter_input[0][0] = 1
print(letter_input)
style_input = np.random.rand(1, 16)

pred_y = decoder.predict([letter_input, style_input])

img = np.clip(pred_y.reshape(40, 24), 0, 1)*255
img = Image.fromarray(np.uint8(img))
img.save('out.png')