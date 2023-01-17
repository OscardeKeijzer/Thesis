# -*- coding: utf-8 -*-

# References:
#     [1] Hoencamp, J., Jain, S., & Kandhai, D. (2022). A Semi-Static 
#         Replication Approach to Efficient Hedging and Pricing of Callable IR 
#         Derivatives. arXiv. doi: 10.48550/arXiv.2202.01027

# Imports
import keras_tuner
import numpy as np
import tensorflow.keras as keras

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Local imports
# ...

class ClipBiases(keras.constraints.Constraint):
    """
    Info:
        This callable class clips the values of the passed biases to the 
        interval [-1, 1].
        
    Input:
        biases: a list containing values of the biases.
        
    Output:
        A list containing the values of the clipped biases.
    """
    def __call__(self,
                 biases: list, 
                 ) -> list:

        return keras.backend.clip(biases, -1., 1.)
    
class BetweenNeg(keras.constraints.Constraint):
    """
    Info:
        This callable class constraints the values of the passed weights to the 
        interval [-1, 0). The method was obtained from the GitHub repository of 
        ref. [1].
                  
        
    Input:
        biases: a list containing values of the weights.
        
    Output:
        A list containing the values of the constrained weights.
    """
    def __call__(self,
                 weights: list
                ) -> list:
        return keras.backend.clip(weights, -1., -0.0001)
            
class BetweenPos(keras.constraints.Constraint):
    """
    Info:
        This callable class constraints the values of the passed weights to the 
        interval (0, 1]. The method was obtained from the GitHub repository of 
        ref. [1].
                  
        
    Input:
        biases: a list containing values of the weights.
        
    Output:
        A list containing the values of the constrained weights.
    """
    def __call__(self, 
                 weights: list
                ) -> list:
        return keras.backend.clip(weights, 0.0001, 1.)

class ShallowFeedForwardNeuralNetwork:
    """
    This class creates an instance of a shallow one-hidden-layer feedforward 
    neural network for use in the Regress Later with Neural Networks (RLNN) 
    pricing method.
    """
    
    def __init__(self,
                 swaption_type: str,
                 activation_func_hidden: str = 'ReLU',
                 activation_func_output: str = 'linear',
                 input_dim: int = 1,
                 n_hidden_nodes: int = 64,
                 n_output_nodes: int = 1,
                 seed_biases: int = None,
                 seed_weights: int = None
                ):
        ## Assign the main neural network parameters from the constructor input
        self.swaption_type = swaption_type
        self.activation_func_hidden = activation_func_hidden
        self.activation_func_output = activation_func_output
        self.input_dim = input_dim
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        
        ## Initialize the input layer of the feed-forward neural network
        self.neural_network = keras.models.Sequential()
        self.neural_network.add(keras.Input(shape=(self.input_dim), 
                                            name='Input_Layer'))
        
        ## Initialize the hidden layer
        # Assign random uniform initializer for values on [-1, 1] and swaption 
        # type-dependent kernel initializer for values on [-1, 0] (receiver 
        # swaption) or on [0, 1] (payer swaption)
        self.init_rand_uniform = keras.initializers.RandomUniform(minval=-1., 
                                                maxval=1., seed=seed_biases)
        if swaption_type.lower() == 'payer':
            self.init_hidden_kernel = keras.initializers.RandomUniform(
                                                minval=.0001, maxval=1., 
                                                seed=seed_weights)
            self.hidden_constraint = BetweenPos()
        elif swaption_type.lower() == 'receiver':
            self.init_hidden_kernel = keras.initializers.RandomUniform(
                                                minval=-1., maxval=-.0001, 
                                                seed=seed_weights)
            self.hidden_constraint = BetweenNeg()
            
        # Construct the hidden layer
        self.neural_network.add(keras.layers.Dense(
                                        activation=self.activation_func_hidden, 
                                        bias_constraint=ClipBiases(), 
                                        bias_initializer=self.init_rand_uniform, 
                                        kernel_constraint=self.hidden_constraint, 
                                        kernel_initializer=self.init_hidden_kernel, 
                                        name='Hidden_Layer', 
                                        units=self.n_hidden_nodes))
        
        ## Construct the output layer
        self.neural_network.add(keras.layers.Dense(
                                        activation=self.activation_func_output, 
                                        kernel_initializer=self.init_rand_uniform, 
                                        name='Output_Layer', 
                                        units=self.n_output_nodes, 
                                        use_bias=False))