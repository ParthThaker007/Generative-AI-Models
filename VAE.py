import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the VAE model
latent_dim = 2  # Dimension of the latent space

# Encoder
encoder_inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation="relu")(decoder_inputs)
decoder_outputs = layers.Dense(784, activation="sigmoid")(x)

# Define the VAE model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

# Define the loss function
reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, vae_outputs)
reconstruction_loss *= 784
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE model
vae.compile(optimizer="adam")

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Train the VAE
vae.fit(x_train, x_train, batch_size=128, epochs=10, validation_data=(x_test, x_test))

# Generate new samples
z_sample = np.random.normal(size=(1, latent_dim))
x_decoded = decoder.predict(z_sample)



# Description:

# The provided code implements a Variational Autoencoder (VAE) model using TensorFlow and Keras. A VAE is a generative model that learns a low-dimensional latent space representation of the input data and can generate new samples from this latent space.

# The VAE model consists of an encoder, a decoder, and a sampling layer that facilitates the reparameterization trick to generate latent space samples.

# The encoder takes input images of shape (784,) and passes them through a dense layer with ReLU activation, resulting in a hidden representation. From this hidden representation, two separate dense layers compute the mean (z_mean) and the logarithm of the variance (z_log_var) of the latent space distribution.

# To facilitate sampling from the latent space, a sampling function is defined that takes in the mean and log variance and generates random samples using the reparameterization trick. It samples from a normal distribution using the given mean and variance.

# The decoder takes the sampled latent space vectors as inputs and passes them through a dense layer with ReLU activation. The output of the decoder is a dense layer with sigmoid activation that generates reconstructed images of shape (784,).

# The VAE model is defined by combining the encoder and decoder models. The encoder model outputs the mean, log variance, and sampled latent vectors. The decoder model takes the sampled latent vectors as inputs and generates the reconstructed images. The VAE model is trained to minimize the reconstruction loss and the KL divergence loss, which captures the divergence between the learned latent space distribution and the standard normal distribution.

# The VAE model is compiled with the Adam optimizer and the loss function that combines the reconstruction loss and KL divergence loss. The binary cross-entropy reconstruction loss is scaled by the number of input dimensions. The model is ready for training.

# The MNIST dataset is loaded and preprocessed by reshaping the images to (784,) and normalizing the pixel values between 0 and 1.

# The VAE is trained on the training set of MNIST images, where the input and target are the same images. The training is performed for a specified number of epochs, with a defined batch size. The validation data is provided to evaluate the model's performance during training.

# After training, the code generates new samples by sampling random latent vectors from a standard normal distribution. The decoder model takes these latent vectors as input and generates corresponding reconstructed images.

# The code demonstrates the implementation and training of a VAE model using TensorFlow and Keras for generating new samples from a learned latent space. It can be extended and customized for different datasets and generative tasks.
