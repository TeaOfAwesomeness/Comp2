# welcome to comp 2
# Imports
import numpy as np
import random as rand

# Input node for the network
class INode:
    val = 0

    # Set node input value
    def set_val(self, in_val):
        if in_val is not None:
            self.val = in_val
            return 1
        else:
            return 0

    # Return last known node input value
    def get_val(self):
        return self.val


# Output node for the network
class ONode:
    # Define a 2D array for storing input values and their associated weights
    inputs = np.array([[], []])

    def __init__(self, number_inputs):
        self.init_weights(self, number_inputs)

    # Initialise random weights for input connections
    def init_weights(self, number_inputs):
        for i in range(0, (number_inputs - 1)):
            random_weight = rand.randrange(1, 100)
            temp_array = np.array([0], [random_weight])  # First is stand in for input value, second is random weight value
            np.append(self.inputs, temp_array, axis=1)  # Nb: python rows and columns are inverted

    # Sets new input values for the node
    def set_input(self, input_number, input_value):
        if self.inputs is not None:
            self.inputs[input_number, 0] = input_value # Access the current row, first column and set to new input value