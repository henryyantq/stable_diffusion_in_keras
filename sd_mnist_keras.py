from keras.datasets import mnist
import numpy as np
from skimage.util import random_noise
from skimage.transform import resize
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, Lambda, Add
from keras.models import Model
import keras.backend as K

# Define the stable diffusion module
def build_stable_diffusion_module(input_shape, filters=64, num_blocks=3):
    input_layer = Input(shape=input_shape)
    x = input_layer

    for i in range(num_blocks):
        # Residual block
        x_skip = x
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda t: t * 0.1)(x)  # scale down the residual
        x = Add()([x_skip, x])  # add the residual

        # Stable diffusion
        x = Lambda(lambda t: t / (K.sqrt(K.mean(K.square(t), axis=-1, keepdims=True)) + K.epsilon()))(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocess the data
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = resize(x_train, (len(x_train), 64, 64, 1))
x_test = resize(x_test, (len(x_test), 64, 64, 1))

# Add random noise to the test images
x_test_noisy = random_noise(x_test, mode='gaussian', var=0.01)

# Define the model
input_shape = (64, 64, 1)
input_layer = Input(shape=input_shape)
stable_diffusion = build_stable_diffusion_module(input_shape)
output = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(stable_diffusion(input_layer))
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(x_train, x_train, batch_size=32, epochs=10, validation_data=(x_test_noisy, x_test))

# Test the model
denoised_images = model.predict(x_test_noisy)

# Plot the original, noisy, and denoised images for a random test example
idx = np.random.randint(len(x_test))
plt.subplot(1, 3, 1)
plt.imshow(x_test[idx].squeeze(), cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(x_test_noisy[idx].squeeze(), cmap='gray')
plt.title('Noisy')
plt.subplot(1, 3, 3)
plt.imshow(denoised_images[idx].squeeze(), cmap='gray')
plt.title('Denoised')
plt.show()
