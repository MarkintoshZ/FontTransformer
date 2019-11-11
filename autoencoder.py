from keras.layers import Dense, Conv2D, Deconvolution2D, \
    MaxPool2D, UpSampling2D, Flatten, Dropout, Reshape,\
    Concatenate
from keras.models import Sequential, Model, load_model, Input
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2

X = []
y_label = []
for path in os.listdir('./datasets'):
    print(path)
    if path == '.DS_Store':
        continue
    for image_path in os.listdir('./datasets/' + path):
        try:
            image = Image.open(os.path.join('./datasets/' + path, image_path))
        except OSError:
            continue

        data = np.asarray(image.convert('L'))
        data = data / 255
        data = np.clip(data, 0, 1)
        assert(data.max() <= 1)
        assert(data.min() >= 0)
        X.append(data)
        y_label.append(image_path[0])

X = np.array(X).reshape(-1, 40, 24, 1)
lb = LabelEncoder()
y_label_transformed = lb.fit_transform(y_label)
y = to_categorical(y_label_transformed)

# encoder
input_img = Input(shape=(40, 24, 1))
x = Conv2D(16, 3, activation='selu', padding='same')(input_img)
x = Conv2D(16, 3, activation='selu', padding='same')(x)
# x = MaxPool2D(2, padding='same')(x)
x = Conv2D(16, 3, activation='selu', padding='same')(x)
# x = MaxPool2D(2, padding='same')(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='selu')(x)
x = Dense(256, activation='selu')(x)
encoded = Dense(26, activation='softmax')(x)
style = Dense(16, activation='sigmoid')(x)

# decoder
x = Concatenate()([encoded, style])
x = Dense(256, activation='selu')(x)
x = Dense(512, activation='selu')(x)
x = Dense(15360)(x)
x = Reshape((40, 24, 16))(x)
# x = UpSampling2D(2)(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
# x = UpSampling2D(2)(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
decoded = Conv2D(1, 3, activation='linear', padding='same')(x)

# input_img = Input(shape=(40, 24, 1))
# x = Flatten()(input_img)
# encoded = Dense(26, activation='softmax')(x)
# decoded = input_img
model = Model(input_img, [encoded, decoded])

model.compile(loss=['categorical_crossentropy', 'mse'],
            optimizer='adam',
            metrics=['accuracy', 'mse'])

model.summary()
model.fit(X, [y, X], batch_size=32, epochs=100, shuffle=True)
model.save('autoencoder_1.h5')
