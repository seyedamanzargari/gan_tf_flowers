import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_images(fake_images):
  plt.imshow(fake_images[0])
  plt.axis('off')
  plt.show()


def generate_images(generator, noise):
  return generator(noise, training=False)

def generate_noise(batch_size, num_latent_inputs):
  return tf.random.normal([batch_size, num_latent_inputs])

def generate_image_from_weights(generator, weights_file):
    # Load the saved weights into the generator
    generator.load_weights(weights_file)

    # Generate a single image using random noise as input
    noise = tf.random.normal([1, config.NUM_LATENT_INPUTS])
    generated_image = generator(noise, training=False)

    # Rescale the pixel values to be between 0 and 255
    generated_image = ((generated_image + 1) / 2.0) * 255

    # Convert the generated image to a NumPy array and cast to uint8
    generated_image = generated_image.numpy()[0].astype(np.uint8)

    # Display the generated image
    plt.imshow(generated_image)
    plt.axis('off')
    plt.show()