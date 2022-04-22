# welcome to comp 2
# Imports
import numpy as np
import random as rand
import math
import tkinter as tk
import csv
import copy


class Main:
    # Global variables
    network = None

    def __init__(self):
        root = tk.Tk()
        frame = tk.Frame(root, height=700, width=700)
        frame.pack()
        self.create_gui(tk, frame)
        root.mainloop()
        self.main()

    def create_gui(self, tk, frame):
        # Command function for btn_gen
        def generate_network():
            # Return text values
            inodes = txt_in.get("1.0", "end-1c")
            onodes = txt_out.get("1.0", "end-1c")
            # Send numeric values to network
            if inodes.isnumeric() and onodes.isnumeric():
                self.network = Network(int(inodes), int(onodes))

        # Command function of btn_train
        def train_network():
            lr = txt_lr.get("1.0", "end-1c")
            if lr.isnumeric():
                input_data = self.read(txt_train.get("1.0", "end-1c"))
                if len(input_data) > 1:
                    self.network.train(input_data, len(input_data), lr)
                    print("train test")

        # Command function of btn_test
        def test_network():
            print("test test")

        # All label widgets
        lbl_in = tk.Label(frame, text="No. Input Nodes").grid(row=0, column=0)
        lbl_out = tk.Label(frame, text="No. Output Nodes").grid(row=0, column=2)
        lbl_gen = tk.Label(frame, text="Generate").grid(row=0, column=4)
        lbl_train = tk.Label(frame, text="Training Data:").grid(row=1, column=0)
        lbl_lr = tk.Label(frame, text="Learning Rate:").grid(row=1, column=2)
        lbl_test = tk.Label(frame, text="Test Data:").grid(row=2, column=0)
        lbl_net = tk.Label(frame, text="Network").grid(row=3, column=1, columnspan=2)

        # All function button widgets
        btn_gen = tk.Button(frame, command=generate_network, width=2).grid(row=0, column=5)
        btn_train = tk.Button(frame, command=train_network, width=2).grid(row=1, column=4)
        btn_test = tk.Button(frame, command=test_network, width=2).grid(row=2, column=2)

        # All text widgets
        txt_in = tk.Text(frame, width=2, height=1)
        txt_in.grid(row=0, column=1)
        txt_out = tk.Text(frame, width=2, height=1)
        txt_out.grid(row=0, column=3)
        txt_train = tk.Text(frame, width=20, height=1)
        txt_train.grid(row=1, column=1)
        txt_lr = tk.Text(frame, width=2, height=1)
        txt_lr.grid(row=1, column=3)
        txt_test = tk.Text(frame, width=20, height=1)
        txt_test.grid(row=2, column=1)

    def main(self):
        print("test")

    # Read csv file data
    def read(self, path):
        file = open(path, newline='')
        csv_data = csv.reader(file)
        input_data = list()
        for row in csv_data:
            input_data.append(row)
        return input_data

    # Function for unit testing
    def test(self):
        # INode unit test
        inode_test1 = INode(0)
        inode_test1.set_val(7)
        print(inode_test1.get_val())

        # ONode unit test
        #onode_test1 = ONode(0, 2)
        #onode_test1.init_weights(2) # Initialise weights, also sets up inputs[]
        #onode_test1.set_input(0, 2) # Test input 1
        #onode_test1.set_input(1, 3) # Test input 2
        #print(onode_test1.get_output()) # Return suitability value


# Contains Kohonen map
class Network:
    num_inodes = 0
    num_onodes = 0
    ls_inodes = list((0, 1))
    ls_onodes = None
    arr_onodes = None
    radius_const = 0
    map_width = 0
    bmu_list = None

    def __init__(self, num_inodes, num_onodes):
        self.num_inodes = num_inodes
        self.num_onodes = num_onodes
        self.map_width = round(math.sqrt(num_onodes))  # Calculate map width
        map_radius = self.map_width / 2  # Initial learning radius
        node_num = 0  # Setup node id

        # Make array of ONodes of size map width by map width
        for i in range(0, self.map_width):
            # Make a temporary array of ONodes of size 1 by map width
            for j in range(0, self.map_width):
                onode = ONode(node_num, i, j, num_inodes)  # New node
                onode.init_weights(num_inodes)  # Randomise weights
                if i == 0 and j == 0:
                    ls_onodes = list([copy.deepcopy(onode)])
                else:
                    ls_onodes.append([copy.deepcopy(onode)])
                node_num += 1

        # Create list of input nodes
        for i in range(0, num_inodes):
            inode = INode(i)
            if i == 0:
                self.ls_inodes = list([inode])
            else:
                self.ls_inodes.append([copy.deepcopy(inode)])
        print("hello world")

    # Train network on 'inputs' data
    def train(self, inputs, time_const, learn_rate):
        time_current = 0

        # Loop through all training data
        for i in range(0, len(inputs)):
            time_current = time_current + 1  # Learning rate decreases over time

            # Load data into input nodes
            current_data_list = inputs[i]
            dataset_name = current_data_list[0]
            for j in range(0, self.num_inodes):
                x=j+1
                if current_data_list[x].isnumeric:
                    self.ls_inodes[j].set_val(int(current_data_list[x]))
            current_best = None

            # Push input data through network
            bmu = self.push(dataset_name)
            # Add BMU to list
            if bmu_list is None:
                bmu_list = [bmu]
            else:
                bmu_list.append(bmu)

            # Update training constants
            new_learn_rate = learn_rate * math.exp(-(time_current/time_const))  # Fraction of previous w/ respect to time
            radius_current = self.radius_const * math.exp(-(time_current/time_const))  # Fraction of previous w/ respect to time

            # Update weights
            for x in range(0, self.map_width):
                for y in range(0, self.map_width):
                    bmu_dist = math.sqrt(pow((x - bmu.x_coord), 2) + pow((y - bmu.ycoord), 2))  # Calculate distance to bmu using pythagoras
                    if bmu_dist <= radius_current:
                        self.arr_onodes([x], [y]).modify_weights(radius_current, new_learn_rate, bmu_dist)
        return bmu_list

    # Using one set of inputs, iterate once over the network
    def push(self, dataset_name):
        # Push data through network to all output nodes
        for x in range(0, self.map_width):
            for y in range(0, self.map_width):
                for j in range(0, self.num_inodes):
                    self.arr_onodes([x], [y]).set_input(j, self.ls_inodes(j).get_val())  # Update input val from INode
                output = self.arr_onodes([x], [y]).get_output()  # Calculate result of weighted inputs
                if output << current_best or current_best is None:
                    current_best = output  # Update BMU
        self.arr_onodes([current_best.x_coord], [current_best.y_coord]).add_bmu(dataset_name)
        return current_best


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
    list_dataset_names = None

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

    def add_bmu(self, dataset_name):
        if self.list_dataset_names is None:
            self.list_dataset_names = list(dataset_name)
        else:
            self.list_dataset_names.append(dataset_name)


if __name__ == "__main__":
    main = Main()
