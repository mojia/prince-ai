from neuralNetwork import neuralNetwork
import numpy as np

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

inputs = [1.0, 0.5, -1.5]
print(n.query(inputs))
