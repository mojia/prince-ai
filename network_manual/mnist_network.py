import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import neuralNetwork

dir = '/Users/xinwang/ai/dataset/mnist'
train_file = 'mnist_train.csv'
test_file = 'mnist_test.csv'

input_nodes = 784
hidden_nodes = 400
output_nodes = 10
learning_rate = 0.1
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


def train_network():
    training_data = []
    with open(os.path.join(dir, train_file), 'r') as file:
        training_data = file.readlines()

    counter = 0
    for record in training_data:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        counter += 1
        if counter % 100 == 0:
            print('progress ' + str(counter * 100.0 / len(training_data)))

        n.train(inputs, targets)


def test_network():
    with open(os.path.join(dir, test_file), 'r') as file:
        data = file.readlines()

        right = 0
        for record in data:
            all_values = record.split(',')
            correct_label = int(all_values[0])

            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            target = n.query(inputs)
            label = np.argmax(target)

            print('DeepLearning value:' + str(label))
            print('actual value:      ' + record[0] + '\n')

            if label == correct_label:
                right += 1

        print('Conclution:' + str(right * 100.0 / len(data)))


epochs = 6
for e in range(epochs):
    train_network()

test_network()
