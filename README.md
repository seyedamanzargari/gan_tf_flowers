# Gan Tensorflow Flowers Dataset

## Dataset
This code defines two functions for loading and augmenting data from the TensorFlow Datasets (TFDS) `tf_flowers` dataset.

The augment_data function takes an image and a label as input, and applies various random transformations to the image, including resizing, horizontal flipping, and changes to brightness and contrast. The function then returns the augmented image and the original label.

The load_data function loads the 'train' split of the `tf_flowers` dataset using the tfds.load function from TFDS. If the augment parameter is set to True, the function applies the augment_data function to each image in the dataset using the map method of the dataset object. If augment is set to False, the function returns the original dataset without any augmentation.

Both functions use TensorFlow operations to manipulate the images and labels, and return TensorFlow Tensor objects. These functions can be used as part of a TensorFlow model training pipeline to load and augment the 'tf_flowers' dataset.

## Networks 
This code defines a Generative Adversarial Network (GAN) architecture using TensorFlow to generate new images. The GAN consists of two neural networks: a generator and a discriminator.

The `create_generator()` function defines the generator neural network architecture using Conv2DTranspose layers, which upsamples the input image. The generator takes a random vector of a fixed size as input, and produces an output image of the desired size.

The `create_discriminator()` function defines the discriminator neural network architecture, which is used to differentiate between real images and generated images. The discriminator is a Convolutional Neural Network (CNN) with several layers, including convolutional layers, batch normalization, and leaky ReLU activation.

The `gan_loss()` function calculates the loss function of the GAN architecture. It takes as input the output of the discriminator for both the real and generated images, and calculates the losses for both the generator and discriminator. The function also randomly flips the labels of the real images to improve the training stability. The output of the function is the losses and scores for the generator and discriminator, which are used to update the model during training.

## utils
This code defines a set of functions for generating and plotting images with a TensorFlow-based generative model.

`plot_images` takes in a set of generated images and displays the first image in the set using matplotlib.

`generate_images` takes in a generator model and a noise vector, and generates a batch of fake images by passing the noise vector through the generator model.

`generate_noise` generates a random noise vector with shape (batch_size, num_latent_inputs).

`generate_image_from_weights` takes in a generator model and the file path to a saved set of generator weights, and generates a single image using random noise as input. The generated image is then rescaled and displayed using matplotlib.

# Training the Model

## Step 1: Config

This code defines various hyperparameters used in a Generative Adversarial Network (GAN) implementation.

- `PROJECTION_SIZE`: This is the size of the feature map in the generator network.
- `NUM_LATENT_INPUTS`: This is the number of random input values fed to the generator network.
- `NUM_FILTERS`: This is the number of filters used in the convolutional layers of the generator and discriminator networks.
- `FILTER_SIZE`: This is the size of the convolutional filters used in the generator and discriminator networks.
- `SCALE`: This is the scaling factor used in the LeakyReLU activation function in the discriminator network.
- `INPUT_SIZE`: This is the size of the input images.
- `DROPOUT_PROB`: This is the probability of dropout applied to the input layer of the discriminator network.
- `LEARNING_RATE`: This is the learning rate used in the optimizer of both the generator and discriminator networks.
- `EPOCHS`: This is the number of epochs the training will run.
- `BATCH_SIZE`: This is the size of the mini-batch used during training.
- `CHECKPOINT_SAVE_INTERVAL`: This is the interval at which the model checkpoints will be saved during training.

## Step 2: Run the trainer

For start training the gan model, just run the following command:`python main.py`

### Explain the training code:
This code is an implementation of a Generative Adversarial Network (GAN) used to generate synthetic images of flowers. The code first defines some configuration parameters such as the size of the projection of the generator, the number of latent inputs, number of filters, filter size, input size, dropout probability, learning rate, epochs, batch size, and checkpoint save interval.

Then, the code loads the **flower dataset**, creates the discriminator and generator models using the functions defined in the `networks.py` module, and defines the optimizers for the generator and discriminator using the _Adam optimizer_ with the specified learning rate.

The code then defines a training step function that performs one step of training for both the generator and discriminator, calculates the loss for both models, and updates their parameters using the calculated gradients.

The code then defines a training loop that iterates over the number of epochs and calls the `train_step()` function on each batch of images from the dataset. After each epoch, the average generator and discriminator loss and score are calculated, and the generator checkpoints are saved at the specified checkpoint save interval.

Finally, the trained generator model is used to generate synthetic images, and the losses and generated images are logged to TensorBoard. The weights for the generator and discriminator models are saved to disk after training is complete.

### Log training process:
In this code, TensorBoard is used to log the training progress of the GAN model. TensorBoard is a tool included in TensorFlow that allows you to visualize and track various aspects of your model training, including metrics, loss values, and image data.

After defining the checkpoint and TensorBoard callbacks, a log directory is created using the following code:

```python
log_dir = 'logs/gan_tf_flowers'
summary_writer = tf.summary.create_file_writer(log_dir)
```
The `summary_writer` is then used to write the loss values and generated images to the TensorBoard log. For example, the following code writes the generator and discriminator loss values to TensorBoard:

```python
with summary_writer.as_default():
     tf.summary.scalar('generator_loss', s_gen_loss, step=epoch)
     tf.summary.scalar('discriminator_loss', s_disc_loss, step=epoch)
     tf.summary.scalar('generator_score', s_gen_score, step=epoch)
     tf.summary.scalar('discriminator_score', s_disc_score, step=epoch)
```

The `tf.summary.scalar` function is used to log scalar values such as loss or accuracy. The `step` parameter specifies the current training step or epoch. Similarly, the following code writes the generated images to TensorBoard:


```python
with summary_writer.as_default():
    tf.summary.image("Generated Image", fake_images, step=epoch)
```

Here, the `tf.summary.image` function is used to log the generated images. The `"Generated Image"` string specifies the name of the image summary, and `fake_images` is the tensor of generated images.

Once the training is complete, `summary_writer.flush()` is called to write the buffered summary data to disk.

Overall, the use of TensorBoard in this code helps to visualize and track the training progress of the GAN model, making it easier to monitor and debug the training process.

The last training process is available in logs directory and can visible by run the following command in terminal:
`tensorboard --logdir logs`