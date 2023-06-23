import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(784, use_bias=False, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))

    return model

# Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))

    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

# Generate images
def generate_images(model, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.show()

# Load MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
buffer_size = train_images.shape[0]
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

# Train the GAN
epochs = 10
train(train_dataset, epochs)




# Project Description: Generative Adversarial Network (GAN) for Image Generation

# This project utilizes a Generative Adversarial Network (GAN) architecture to generate realistic images that resemble the MNIST dataset of handwritten digits. The GAN consists of two main components: a generator and a discriminator.

# The generator model is responsible for generating synthetic images from random noise. It comprises multiple layers of dense and batch normalization units, followed by activation functions such as LeakyReLU and a final tanh activation. The generated images are reshaped to match the dimensions of the MNIST dataset.

# The discriminator model acts as a binary classifier, distinguishing between real and generated images. It takes in input images, either real or synthetic, and processes them through layers of flattening, dense, batch normalization, and LeakyReLU units. The final output is a single neuron representing the probability of the input being a real image.

# During the training process, the generator and discriminator models are alternately optimized. The generator aims to generate images that fool the discriminator, while the discriminator aims to correctly classify real and generated images. The loss functions used are binary cross-entropy, and the Adam optimizer is employed for both models.

# The training loop runs for a specified number of epochs, iterating through the MNIST dataset in batches. In each training step, random noise is passed through the generator to produce synthetic images. The discriminator then evaluates both real and generated images, and the gradients of the generator and discriminator losses are computed. These gradients are used to update the trainable variables of their respective models via the optimizer.

# Once the GAN is trained, the generate_images() function can be used to produce new images. It takes a random input vector and passes it through the generator to generate a set of synthetic images. These generated images are then displayed using matplotlib.

# The MNIST dataset is loaded, preprocessed, and split into a training set. The images are normalized to a range of -1 to 1. The dataset is shuffled, and a batch size is defined for efficient training.

# To execute the project, the train() function is called with the prepared training dataset and the desired number of epochs. After training, the generate_images() function is invoked with a random input vector to generate a set of new images that resemble handwritten digits. These images are displayed using matplotlib.

# This project showcases the power of GANs in generating realistic images and demonstrates their potential applications in various fields such as computer vision, creative art, and data augmentation.

# # Generate and display new images
# num_examples_to_generate = 16
# random_vector_for_generation = tf.random.normal([num_examples_to_generate, 100])
# generate_images(generator, random_vector_for_generation)
