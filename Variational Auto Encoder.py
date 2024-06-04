import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt


directory_path = r'directoryhere'

def load_and_resize_images_from_folder(folder, target_height, target_width):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            if img is not None:
                img = img.resize((target_width, target_height)) 
                img = np.array(img)
                images.append(img)
    return images

height = 64
width = 64

football_shirts = load_and_resize_images_from_folder(directory_path, height, width)

latent_dim = 256

# Encoder
encoder_inputs = keras.Input(shape=(height, width, 3))
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 512, activation="relu")(decoder_inputs)  # Adjust the input size
x = layers.Reshape((8, 8, 512))(x)  # Adjust the reshape size
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)


encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

#loss function
beta = 0.01
def vae_loss(y_true, y_pred):
    encoder_outputs = encoder(y_true)
    z_mean, z_log_var = encoder_outputs[0], encoder_outputs[1]
    reconstruction_loss = keras.losses.binary_crossentropy(tf.keras.backend.flatten(y_true), tf.keras.backend.flatten(y_pred))
    reconstruction_loss *= height * width
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return tf.reduce_mean(reconstruction_loss + kl_loss)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=vae_loss)

football_shirts = np.array(football_shirts) / 255.0  # Normalize pixel values

train_size = int(0.8 * len(football_shirts))
train_images = football_shirts[:train_size]
val_images = football_shirts[train_size:]

#amend epoch cycles
history = vae.fit(train_images, train_images,
                  epochs=50,
                  batch_size=32,
                  validation_data=(val_images, val_images),
                  callbacks=[
                      keras.callbacks.LambdaCallback(
                          on_epoch_begin=lambda epoch, logs: print("Epoch:", epoch)
                      )
                  ])


loss = vae.evaluate(val_images, val_images)
print("Validation Loss:", loss)

num_samples = 10
latent_samples = np.random.normal(size=(num_samples, latent_dim))
generated_images = decoder.predict(latent_samples)


plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(2, num_samples // 2, i + 1)
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.show()
