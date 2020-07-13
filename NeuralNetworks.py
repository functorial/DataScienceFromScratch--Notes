#
# Neural Networks
#

# Idea: 
#   A directed graph of 'neurons'.
#       When a neuron recieves a set of inputs,
#       it does a calculation which
#       determines if it should fire an output or not.
#       
#       Presumably, the neurons are stimulated from an external source (data)

# Pros:
#   (*) Can solve a wise variety of problems
#       e.g. handwriting recognition, face detection
#   (*) Trendy
#       e.g. used in deep learning
#   
# Cons:
#   (*) Operate as 'black boxes'
#   (*) Can be hard to train large neural nets


# Perceptrons
#   The simplest neural network:
#       One neuron with n binary inputs
#       Computes a weighted sum and fires if the sum is >0
#
#       Weighted_sum = dot(weights, inputs) + bias
#
#       The computation splits the input space into two via the hyperplane defined by
#           dot(weights, inputs) + bias == 0
#       and returns which half space the input is in.
import numpy as np
from scipy.special import expit
from typing import List

def step_function(x: float) -> float:
    return 1. if x >= 0. else 0.

def perceptron_output(weights: List[float], inputs: List[float], bias: float=1) -> float:
    weights = np.array(weights)
    inputs = np.array(inputs)
    return step_function(np.dot(weights, inputs) + bias)

# With properly chosen weights, perceptions can solve many simple problems
# e.g. can make AND and OR gates:

or_weights = [1., 1.]
or_bias = -0.5
assert perceptron_output(or_weights, [1., 1.], or_bias) == 1.
assert perceptron_output(or_weights, [1., 0.], or_bias) == 1.
assert perceptron_output(or_weights, [0., 1.], or_bias) == 1.
assert perceptron_output(or_weights, [0., 0.], or_bias) == 0.

and_weights = [1., 1.]
and_bias = -1.5
assert perceptron_output(and_weights, [1., 1.], and_bias) == 1. 
assert perceptron_output(and_weights, [1., 0.], and_bias) == 0. 
assert perceptron_output(and_weights, [0., 1.], and_bias) == 0. 
assert perceptron_output(and_weights, [0., 0.], and_bias) == 0. 

not_gate_weights = [-1]
not_gate_bias = 0.5
assert perceptron_output(not_gate_weights, [1], not_gate_bias) == 0
assert perceptron_output(not_gate_weights, [0], not_gate_bias) == 1

# NOTE: you cannot build a XOR gate from a perceptron.
#           easy to see from geometry:
#               there does not exist a hyperplane 
#               which separates XOR_True = {(1, 0), (0,1)} and XOR_False = {(1, 1), (0, 0)}


# NOTE: the step function is not differentiable
#       we may want to replace it with a smooth approximation, for calculus purposes
#           expit == the logistic, or sigmoid, function
def neuron_output(weights: List[float], 
                  inputs: List[float], 
                  bias: float=1,
                  activation_function: callable=expit) -> float:
    weights = np.array(weights)
    inputs = np.array(inputs)
    return activation_function(np.dot(weights, inputs))

# NOTE: the topology of the brain is complicated,
#       we will approximate it using a directed graph 
#       with connected COMPLETE bipartite layers
#           This is called a /feed-forward/ neural network
#       e.g.
#           0 --------> 0 ____
#                             \
#           0 --------> 0 ----->>> 0
#                             /     \
#           0 --------> 0 ___/       \               NOTE: Actually, picture is not accurate
#                                     >> 0                  (*) bipartite layer connections 
#           0 --------> 0 ____       /                          are not complete
#                             \     /
#           0 --------> 0 ----->>> 0
#                             /
#           0 --------> 0 ___/

# Represent this neural network as ff_net = Layers[Neurons[Weights]] = List[List[np.ndarray]]
# NOTE: Can simplify the representation:
#               the input layer is a vector
#               each layer is a matrix
#               feed through by matrix multiplication
#               add bias where needed & activate

# ff stands for 'feed-forward', which indicates the shape 
def feed_forward(ff_net: List[List[np.ndarray]],
                 input_vector: np.ndarray,
                 bias: float=1,
                 activation_function: callable=expit) -> np.ndarray:
    """Feeds the input vector through the network. 
    Returns the outputs of all layers"""
    input_vector = np.array(input_vector)
    activate_output = np.vectorize(activation_function)
    layer_outputs = []
    for layer in ff_net:
        input_with_bias = np.append(input_vector, bias)
        layer = np.array(layer)
        layer_output = activate_output( input_with_bias @ layer.T )
        # layer_output = [neuron_output(neuron_weights, input_with_bias, activation_function)
        #                      for neuron_weights in layer]
        layer_outputs.append(layer_output)
        input_vector = layer_output     # this replaces the need for recursion
    
    return layer_outputs


# Build a XOR gate:                                             Scale up weights so that outputs close to 0 or 1
xor_network = [# hidden layer
               [[2., 2, -1],    # OR gate               
               [2., 2, -3]],    # AND gate
               # output layer
               [[6., -6, -3]]]    # OR but not AND

print(f"True: {feed_forward(xor_network, [1, 0])}")
print(f"True: {feed_forward(xor_network, [0, 1])}")
print(f"False: {feed_forward(xor_network, [0, 0])}")
print(f"False: {feed_forward(xor_network, [1, 1])}")
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001


