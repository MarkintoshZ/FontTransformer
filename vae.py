from keras.layers import Dense, Conv2D, Deconvolution2D, \
    MaxPool2D, UpSampling2D, Flatten, Dropout, Reshape,\
    Concatenate, Lambda
from keras.models import Sequential, Model, load_model, Input
from keras.losses import mse, categorical_crossentropy
from keras.utils import to_categorical
from keras import backend as K
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


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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
z_mean = Dense(16, name='z_mean')(x)
z_log_var = Dense(16, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(16,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(input_img, [encoded, z_mean, z_log_var, z], name='encoder')
encoder.summary()

# decoder
alphabet_inputs = Input(shape=(26,))
latent_inputs = Input(shape=(16,), name='z_sampling')
x = Concatenate()([alphabet_inputs, latent_inputs])
x = Dense(256, activation='selu')(x)
x = Dense(512, activation='selu')(x)
x = Dense(15360)(x)
x = Reshape((40, 24, 16))(x)
# x = UpSampling2D(2)(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
# x = UpSampling2D(2)(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
x = Conv2D(16, 2, activation='selu', padding='same')(x)
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model([alphabet_inputs, latent_inputs], decoded, name='decoder')
decoder.summary()

# instantiate VAE model
encoder_out = encoder(input_img)
outputs = decoder([encoder_out[0], encoder_out[3]])
vae = Model(input_img, [encoder_out[0], outputs], name='vae_mlp')


def custom_loss(y_true, y_pred):
    reconstruction_loss = mse(input_img, y_pred[1])
    reconstruction_loss *= 960
    reconstruction_loss = K.sum(reconstruction_loss, axis=-1)
    classification_loss = categorical_crossentropy(y_true, y_pred[0])
    classification_loss = K.sum(classification_loss, axis=-1)
    kl_loss = 1 + K.mean(z_log_var) - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss + classification_loss)
    return vae_loss

# vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=custom_loss)
vae.summary()

vae.fit(X, [y, X], batch_size=32, epochs=100, shuffle=True)
vae.save('vae.h5')
