from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class FullyConnected(Layer):
    """
    FullyConnected is a custom neural network layer that constructs a sequence of fully connected layers 
    with optional batch normalization and dropout.

    Attributes:
        n_fc (int): The number of fully connected layers to create.
        hidden_phi (int): The number of units in each hidden layer (except the output layer).
        out_size (int): The number of units in the output layer.
        final_activation (str or callable): The activation function for the output layer.
        name (str): The base name for the layers.
        kernel_reg (tf.keras.regularizers.Regularizer): Regularizer function applied to the kernel weights.
        kernel_init (str or callable): Initializer for the kernel weights.
        activation (str or callable, optional): Activation function for the hidden layers. Defaults to 'elu'.
        bias_initializer (str or callable, optional): Initializer for the bias vector. Defaults to None.
        dropout (bool, optional): Whether to apply dropout after each hidden layer. Defaults to False.
        batch_norm (bool, optional): Whether to apply batch normalization before each hidden layer. Defaults to False.
        use_bias (bool, optional): Whether the dense layers use a bias vector. Defaults to True.
        dropout_rate (float, optional): The dropout rate, between 0 and 1. Defaults to 0.0.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Methods:
        call(x):
            Applies the sequence of layers to the input tensor `x`.

    Example:
        # Create a FullyConnected layer with 3 hidden layers, 64 units each, and ReLU activation
        fc_layer = FullyConnected(
            n_fc=4,
            hidden_phi=64,
            out_size=10,
            final_activation='softmax',
            name='fc_layer',
            kernel_reg=None,
            kernel_init='he_normal',
            activation='relu',
            dropout=True,
            dropout_rate=0.5,
            batch_norm=True
        )
    """
    def __init__(self, n_fc, hidden_phi, out_size,  final_activation, name, kernel_reg, kernel_init, activation='elu',
                 bias_initializer=None, dropout=False, batch_norm=False, use_bias=True,  dropout_rate=0.0, **kwargs):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_fc-1):
            if batch_norm:
                self.Layers.append(BatchNormalization())
            self.Layers.append(Dense(units=hidden_phi, activation=activation, kernel_initializer=kernel_init,
                                     bias_initializer=bias_initializer, use_bias=use_bias,
                                     kernel_regularizer=kernel_reg, name=name + str(i)))
            if dropout:
                self.Layers.append(Dropout(dropout_rate))
        self.Layers.append(Dense(units=out_size, activation=final_activation, name=name + 'out'))


    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x


