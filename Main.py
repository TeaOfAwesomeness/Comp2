# welcome to comp 2
# Imports
import numpy as np
import random as rand
import math


class Main:
    def __init__(self):
       self.test()

    # Function for unit testing
    def test(self):
        # INode unit test
        inode_test1 = INode(0)
        inode_test1.set_val(7)
        print(inode_test1.get_val())

        # ONode unit test
        onode_test1 = ONode(0, 2)
        onode_test1.init_weights(2) # Initialise weights, also sets up inputs[]
        onode_test1.set_input(0, 2) # Test input 1
        onode_test1.set_input(1, 3) # Test input 2
        print(onode_test1.get_output()) # Return suitability value


class Network:
    inodes = None
    onodes = None
    arr_onodes = None
    radius_const = 0
    map_width = 0

    def __init__(self, num_inodes, num_onodes):
        self.map_width = round(math.sqrt(num_onodes))  # Calculate map width
        map_radius = self.map_width / 2  # Initial learning radius
        node_num = 0  # Setup node id

        # Make array of ONodes of size map width by map width
        for i in range(0, self.map_width):
            # Make a temporary array of ONodes of size 1 by map width
            for j in range(0, self.map_width):
                onode = ONode(node_num)  # New node
                onode.x_coord = i
                onode.y_coord = j
                onode.init_weights(num_inodes)  # Randomise weights
                if j == 0:
                    temp_arr = np.array(onode)
                else:
                    temp_arr.append(onode)
                node_num += 1

            # Transpose temporary array to make it size: map width by 1, then append to main ONode array
            if i == 0:
                arr_onodes = np.array(np.transpose(temp_arr))
            else:
                arr_onodes.append(np.transpose(temp_arr))

        # Create list of input nodes
        self.inodes = list(INode)
        for i in range(0, num_inodes):
            inode = INode(i)
            self.inodes.append(inode)

        # Create list of output nodes -----OLD list now an array
        #self.onodes = list(ONode)
        #for i in range(0, num_onodes):
        #    onode = ONode(i)  # New node
        #    onode.init_weights(num_inodes)  # Randomise weights
        #    self.onodes.append(onode)  # Add to list

    def train(self, inputs, time_const, learn_rate):
        time_current = 0

        # Loop through all training data
        for i in range(0, inputs.shape[0]):
            time_current = time_current + 1  # Learning rate decreases over time

            # Load data into input nodes
            for j in range(0, len(self.inodes)):
                self.inodes(j).set_val(inputs([j], [i]))
            current_best = None

            # Push data through network to all output nodes ----OLD uses list
            #for j in range(0, len(self.onodes)):
            #    for x in range(0, len(self.inodes)):
            #        self.onodes(j).set_input(x, self.inodes(x).get_val())
            #    output = self.onodes(j).get_output()  # Calculate result of input nodes
            #    if output << current_best or current_best is None:
            #        current_best = output  # Update BMU

            # Push data through network to all output nodes
            for x in range(0, self.map_width):
                for y in range(0, self.map_width):
                    for j in range(0, len(self.inodes)):
                        self.arr_onodes([x], [y]).set_input(j, self.inodes(j).get_val())  # Update input val from INode
                    output = self.arr_onodes([x], [y]).get_output()  # Calculate result of weighted inputs
                    if output << current_best or current_best is None:
                        current_best = output  # Update BMU

            # Update training constants
            new_learn_rate = learn_rate * math.exp(-(time_current/time_const))  # Fraction of previous w/ respect to time
            radius_current = self.radius_const * math.exp(-(time_current/time_const))  # Fraction of previous w/ respect to time

    def push(self):
        # Push data through network to all output nodes
        for x in range(0, self.map_width):
            for y in range(0, self.map_width):
                for j in range(0, len(self.inodes)):
                    self.arr_onodes([x], [y]).set_input(j, self.inodes(j).get_val())  # Update input val from INode
                output = self.arr_onodes([x], [y]).get_output()  # Calculate result of weighted inputs
                if output << current_best or current_best is None:
                    current_best = output  # Update BMU


# Input node for the network
class INode:
    val = 0

    def __init__(self, node_num):
        self.id = node_num

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
    # Coordinates in hypothetical matrix
    x_coord = 0
    y_coord = 0

    def __init__(self, node_num, node_x, node_y, number_inputs):
        self.id = node_num
        self.x_coord = node_x
        self.y_coord = node_y
        self.init_weights(number_inputs)  # Randomise initial weights

    # Initialise random weights for input connections
    def init_weights(self, number_inputs):
        for i in range(0, number_inputs):
            random_weight = rand.randrange(1, 100)
            temp_array = np.array([[0], [random_weight]])  # First is stand in for input value, second is random weight value
            self.inputs = np.append(self.inputs, temp_array, axis=1)  # Nb: python rows and columns are inverted

    # Sets new input values for the node
    def set_input(self, input_number, input_value):
        if self.inputs is not None:
            self.inputs[input_number, 0] = input_value # Access the current row, first column and set to new input value

    # Returns a float representing suitability of node as BMU
    def get_output(self):
        result = 0
        for i in range(0, self.inputs.shape[0]):
            result = result + math.pow((self.inputs[i, 0] - self.inputs[i, 1]), 2) # Summates all squared differences between input value and weight
        result = math.sqrt(result)
        return result

    # Train weights according to new BMU
    def modify_weights(self, radius_current, learning_rate, bmu_dist):
        dist_effect = math.exp(-(math.pow(bmu_dist, 2)/(2*radius_current)))
        for i in range(self.inputs.shape[0]):
            temp_val_desired = self.inputs[i, 0]
            temp_weight = self.inputs[i, 1]
            self.inputs[i, 1] = temp_weight + (dist_effect*learning_rate*(temp_val_desired - temp_weight))


if __name__ == "__main__":
    main = Main()
