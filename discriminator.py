from keras.layers import (
    Conv2D,
    Flatten,
    BatchNormalization,
    Dense,
    Activation
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from typing import Tuple, Union


class Discriminator:
    def __init__(self,
                 kernel_size: Tuple,
                 stride_size: Tuple,
                 first_layer_filter: Union[int, float],
                 image_shape,
                 lr: float = 0.0002,
                 beta: float = 0.5,
                 loss: str = 'binary_crossentropy',
                 activation: str = 'sigmoid'):
        self.image_shape = image_shape
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.first_layer_filter = first_layer_filter
        self.second_layer_filter = first_layer_filter * 2
        self.third_layer_filter = self.second_layer_filter * 2
        self.last_layer_size = self.third_layer_filter * 2
        self.activation = activation
        self.lr = lr
        self.beta = beta
        self.loss = loss

    def discriminator(self):
        """
        The function takes training_images and outputs a vector of probabilities (btw 0 and 1) that the training_images are real using a convolution neural net.
        The parameters that need to be passed are:
        * input: set of training_images
        * Training: False or True, corresponding to whether the net is still on training or not
        * Reuse: Set to default to False, If True the function will reuse the variables already created
        """

        # Initializate the neural network
        discriminator = Sequential()

        # Convolution, bias, activate
        discriminator.add(Conv2D(filters=self.first_layer_filter,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride_size,
                                 padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform',
                                 input_shape=self.image_shape))
        # Activate
        discriminator.add(LeakyReLU(0.2))
        # Convolution
        discriminator.add(Conv2D(filters=self.second_layer_filter,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride_size,
                                 padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))

        # Normalize
        discriminator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        discriminator.add(LeakyReLU(0.2))

        # Convolution
        discriminator.add(Conv2D(filters=self.third_layer_filter,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride_size,
                                 padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))

        # Normalize
        discriminator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        discriminator.add(LeakyReLU(0.2))
        # Convolution
        discriminator.add(Conv2D(filters=self.last_layer_size,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride_size,
                                 padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        # Normalize
        discriminator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        optimizer = Adam(lr=self.lr, beta_1=self.beta)
        discriminator.compile(loss=self.loss,
                              optimizer=optimizer,
                              metrics=None)

        return discriminator
