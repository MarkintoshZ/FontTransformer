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

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def train_generater():
    fonts_dir = os.listdir('./datasets')
    fonts_dir.remove('.DS_Store')
    while True:
        input_img, classfication_output, \
            output_img, classfication_input = [], [], [], []
        for _ in range(32):
            font = fonts_dir[np.random.randint(8)]
            in_img, clas_out = train_sample(font)
            out_img, clas_in = train_sample(font)
            input_img.append(in_img)
            classfication_input.append(clas_in)
            output_img.append(out_img)
            classfication_output.append(clas_out)
        input_img = np.array(input_img)
        classfication_output = np.array(classfication_output)
        output_img = np.array(output_img)
        classfication_input = np.array(classfication_input)
        
        yield [input_img, classfication_input], \
              [classfication_output, output_img]

def train_sample(font):
    input_idx = np.random.randint(26)
    letter_index = np.zeros((26,))
    letter_index[input_idx] = 1
    img = Image.open('datasets/{}/{}.png'.format(font, alphabet[input_idx]))
    img = np.asarray(img.convert('L')) / 255
    img = np.expand_dims(img, -1)
    return img, letter_index


# encoder
input_img = Input(shape=(40, 24, 1))
x = Conv2D(16, 3, activation='selu', padding='same')(input_img)
x = Conv2D(16, 3, activation='selu', padding='same')(x)
x = Conv2D(16, 3, activation='selu', padding='same')(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='selu')(x)
x = Dense(256, activation='selu')(x)
encoded = Dense(26, activation='softmax')(x)
style = Dense(7, activation='sigmoid')(x)

# decoder
encoded_input = Input(shape=(26,))
style_input = Input(shape=(7,))
x = Concatenate()([encoded_input, style_input])
x = Dense(256, activation='selu')(x)
x = Dense(512, activation='selu')(x)
x = Dense(15360)(x)
x = Reshape((40, 24, 16))(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
decoded = Conv2D(1, 3, activation='linear', padding='same')(x)

encoder = Model(input_img, [encoded, style])
decoder = Model([encoded_input, style_input], decoded)
decoder_out = decoder([encoded_input, encoder(input_img)[1]])
autoencoder = Model([input_img, encoded_input], [encoded, decoder_out])

autoencoder.compile(loss=['categorical_crossentropy', 'mse'],
            optimizer='adam',
            metrics=['accuracy', 'mse'])

autoencoder.summary()
autoencoder.fit_generator(train_generater(), steps_per_epoch=512, epochs=3)

autoencoder.save('models/autoencoder_2.h5')
encoder.save('models/encoder_2.h5')
decoder.save('models/decoder_2.h5')