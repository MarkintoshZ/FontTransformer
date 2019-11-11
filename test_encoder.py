from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report
import os

model = load_model('models/encoder_2.h5')

image = Image.open('datasets/Courier New Italic/a.png')
data = np.asarray(image.convert('L'))
data = data / 255

X = data.reshape(-1, 40, 24, 1)

pred_y = model.predict(X)[0]
alphabet = 'abcdefghijklmnopqrstuvwxyz'
res = alphabet[np.argmax(pred_y)]

print(res)
