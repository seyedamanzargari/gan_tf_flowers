import tensorflow as tf
import config

def create_generator():
  layers_generator = [
      tf.keras.layers.Input(shape=(config.NUM_LATENT_INPUTS)),
      tf.keras.layers.Dense(units=tf.reduce_prod(config.PROJECTION_SIZE)),
      tf.keras.layers.Reshape(target_shape=config.PROJECTION_SIZE),
      tf.keras.layers.Conv2DTranspose(filters=4*config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=1, padding='valid'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2DTranspose(filters=2*config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2DTranspose(filters=config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.Activation('tanh')
  ]

  generator = tf.keras.Sequential(layers_generator)
  return generator

def create_discriminator():
  layers_discriminator = [
      tf.keras.layers.Input(shape=config.INPUT_SIZE),
      tf.keras.layers.Dropout(rate=config.DROPOUT_PROB),
      tf.keras.layers.Conv2D(filters=config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.LeakyReLU(alpha=config.SCALE),
      tf.keras.layers.Conv2D(filters=2*config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(alpha=config.SCALE),
      tf.keras.layers.Conv2D(filters=4*config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(alpha=config.SCALE),
      tf.keras.layers.Conv2D(filters=8*config.NUM_FILTERS, kernel_size=config.FILTER_SIZE, strides=2, padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(alpha=config.SCALE),
      tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1),
      tf.keras.layers.Activation('sigmoid')
  ]

  discriminator = tf.keras.Sequential(layers_discriminator)
  return discriminator

def gan_loss(y_real, y_generated):
    
    # Calculate the score of the discriminator.
    score_d = (tf.reduce_mean(y_real) + tf.reduce_mean(1-y_generated)) / 2

    # Calculate the score of the generator.
    score_g = tf.reduce_mean(y_generated)

    # Randomly flip the labels of the real images.
    num_observations = tf.shape(y_real)[0]
    idx = tf.random.uniform([num_observations], maxval=1.0) < 0.35
    y_real = tf.where(idx, 1 - y_real, y_real)

    # Calculate the loss for the discriminator network.
    loss_d = -tf.math.reduce_mean(tf.math.log(y_real)) - tf.math.reduce_mean(tf.math.log(1 - y_generated))

    # Calculate the loss for the generator network.
    loss_g = -tf.math.reduce_mean(tf.math.log(y_generated))

    return loss_g, loss_d, score_g, score_d