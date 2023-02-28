import networks
import config
import tensorflow as tf
import tqdm
import utils
from dataset import load_data
import os
import cv2

#os.makedirs('out_images', exist_ok=True)
os.makedirs('weights', exist_ok=True)

dataset = load_data(True)

# Load Models
discriminator = networks.create_discriminator()
generator = networks.create_generator()

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LERNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LERNING_RATE)

# Define the checkpoint and TensorBoard callbacks
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


log_dir = 'logs/gan_tf_flowers'
summary_writer = tf.summary.create_file_writer(log_dir)

@tf.function
def train_step(images):
    # Generate noise for the generator
    noise = tf.random.normal([config.BATCH_SIZE, config.NUM_LATENT_INPUTS])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images using the generator
        generated_images = generator(noise, training=True)

        # Get the discriminator's predictions for the real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate the generator and discriminator losses
        gen_loss, disc_loss, score_g, score_d = networks.gan_loss(real_output, fake_output)

    # Calculate the gradients and update the generator and discriminator parameters
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, score_g, score_d


# Define the training loop
for epoch in range(config.EPOCHS):
    # Print the progress of the training
    print(f'Epoch {epoch+1}/{config.EPOCHS}')
    s_gen_loss = []
    s_disc_loss = []
    s_gen_score = []
    s_disc_score = []
    for images, _ in tqdm.tqdm(dataset.batch(config.BATCH_SIZE)):
        # Scale the pixel values of the images to be between -1 and 1
        images = tf.cast(images, tf.float32)

        # Train the generator and discriminator
        gen_loss, disc_loss, score_g, score_d = train_step(images)

        s_gen_loss.append(gen_loss)
        s_disc_loss.append(disc_loss)

        s_gen_score.append(score_g)
        s_disc_loss.append(disc_loss)


    s_gen_loss = tf.math.reduce_mean(s_gen_loss)
    s_disc_loss = tf.math.reduce_mean(s_disc_loss)

    s_gen_score = tf.math.reduce_mean(s_gen_score)
    s_disc_score = tf.math.reduce_mean(s_disc_score)

    # Save the generator and discriminator checkpoints
    if (epoch + 1) % config.CHECKPOINT_SAVE_INTERVAL == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    fake_images = utils.generate_images(generator, utils.generate_noise(1, config.NUM_LATENT_INPUTS))
    #utils.plot_images(fake_images)

    # Log the losses to TensorBoard
    with summary_writer.as_default():
         tf.summary.scalar('generator_loss', s_gen_loss, step=epoch)
         tf.summary.scalar('discriminator_loss', s_disc_loss, step=epoch)
         tf.summary.scalar('generator_score', s_gen_score, step=epoch)
         tf.summary.scalar('discriminator_score', s_disc_score, step=epoch)
         tf.summary.image("Generated Image", fake_images, step=epoch)

    #cv2.imwrite(f"out_images/{epoch}.jpg", fake_images[0].numpy()*255)

    if (epoch + 1) % 50 == 0:
        generator_weights_path = f'weights/generator_weights_{epoch+1}.h5'
        discriminator_weights_path = f'weights/discriminator_weights_{epoch+1}.h5'
        generator.save_weights(generator_weights_path)
        discriminator.save_weights(discriminator_weights_path)

# Define the paths for the saved weights
generator_weights_path = 'weights/generator_weights_last.h5'
discriminator_weights_path = 'weights/discriminator_weights_last.h5'

# Save the weights
generator.save_weights(generator_weights_path)
discriminator.save_weights(discriminator_weights_path)

summary_writer.flush()