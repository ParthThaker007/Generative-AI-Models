import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from scipy.stats import norm

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Define the VAE model
input_shape = x_train.shape[1:]
latent_dim = 128

# Encoder
inputs = Input(shape=input_shape)
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
latent_inputs = Input(shape=(latent_dim,))
x = Dense(8 * 8 * 64, activation='relu')(latent_inputs)
x = Reshape((8, 8, 64))(x)
x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

# VAE model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(latent_inputs, outputs, name='decoder')

# Connect encoder and decoder
outputs = decoder(encoder(inputs)[2])

# VAE model
vae = Model(inputs, outputs, name='vae')

# Define the loss function
reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the model
vae.compile(optimizer='adam')

# Train the VAE model
vae.fit(x_train, epochs=10, batch_size=12, validation_data=(x_test, None))

# Generate images from random samples in the latent space
n_samples = 10
random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
decoded_images = decoder.predict(random_latent_vectors)

# Display the generated images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
for i in range(n_samples):
    ax = plt.subplot(2, n_samples // 2, i + 1)
    plt.imshow(decoded_images[i])
    plt.axis('off')
plt.show()
