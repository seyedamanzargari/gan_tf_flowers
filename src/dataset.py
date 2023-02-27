import tensorflow_datasets as tfds
import tensorflow as tf

def augment_data(image, label):
    image = tf.image.resize(image, [64, 64])
    image = (image / 255.0)
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    # Randomly adjust the brightness of the image
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Randomly adjust the contrast of the image
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # Return the augmented image and label
    return image, label

def load_data(augment=True):
    dataset = tfds.load(
        'tf_flowers',
        split=['train'],
        as_supervised=True,
    )[0]

    if augment:
        return dataset.map(augment_data)
    else:
        return dataset