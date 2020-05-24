from typing import Tuple, Union
import numpy as np

from keras.layers import (
    Conv2DTranspose,
    Reshape,
    BatchNormalization,
    Dense,
    Activation
)
from keras.models import Sequential
from keras.optimizers import Adam


class Generator:
    def __init__(self,
                 initial_dimensions: Tuple,
                 reshape_dimensions: Tuple,
                 kernel_size: Tuple,
                 stride_size: Tuple,
                 output_channels: Union[int, float],
                 lr: float = 0.00015,
                 beta: float = 0.5,
                 loss: str = 'binary_crossentropy'):
        """
        The parameters that need to be passed are:
        initial_dimensions: A tuple indicating the dimensions of the noise vector
        reshape_dimensions: The dimension in which the initial vector is reshaped
        kernel: The size of the kernel
        stride: The size of how the window is slided
        output_channel: If we want a RGB color image, then 3, if grey then 1
        lr: the learning rate
        beta: the beta factor
        loss: The name of the loss function to calculate how the generator is learning.
        """
        self.initial_dimensions = initial_dimensions
        self.reshape_dimensions = reshape_dimensions
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.first_layer_filter = int(self.reshape_dimensions[2]/2)
        self.second_layer_filter = int(self.first_layer_filter/2)
        self.third_layer_filter = int(self.second_layer_filter/2)
        self.output_channels = output_channels
        self.lr = lr
        self.beta = beta
        self.loss = loss

    def generator(self):
        """
        The function generates an training_images through a deconvolution neural net.
        """

        # Define type of neural network. Would be sequential.
        generator = Sequential()
        # Layer1. Now we project and reshape.
        generator.add(Dense(units=np.prod(self.reshape_dimensions),
                            kernel_initializer='glorot_normal',
                            input_shape=self.initial_dimensions))
        # Reshape
        generator.add(Reshape(target_shape=self.reshape_dimensions))
        # Normalize
        generator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activation
        generator.add(Activation('relu'))
        # Now we need to add layers. convolution, bias, activate.
        # Convolution
        generator.add(Conv2DTranspose(filters=self.first_layer_filter,
                                      kernel_size=self.kernel_size,
                                      strides=self.stride_size,
                                      padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_normal'))
        # Bias
        generator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        generator.add(Activation('relu'))
        # Convolution
        generator.add(Conv2DTranspose(filters=self.second_layer_filter,
                                      kernel_size=self.kernel_size,
                                      strides=self.stride_size,
                                      padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_normal'))
        # Bias
        generator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        generator.add(Activation("relu"))
        # Convolution
        generator.add(Conv2DTranspose(filters=self.third_layer_filter,
                                      kernel_size=self.kernel_size,
                                      strides=self.stride_size,
                                      padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_normal'))

        # Bias
        generator.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
        # Activate
        generator.add(Activation('relu'))

        # Last layer. Convolution
        generator.add(Conv2DTranspose(filters=self.output_channels,
                                      kernel_size=self.kernel_size,
                                      strides=self.stride_size,
                                      padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_normal'))
        # Activate
        generator.add(Activation('tanh'))

        optimizer = Adam(lr=self.lr, beta_1=self.beta)
        generator.compile(loss=self.loss,
                          optimizer=optimizer,
                          metrics=None)

        return generator
