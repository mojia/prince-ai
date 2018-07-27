from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import utils
import os
import numpy as np
import time

dir = '/Users/xinwang/ai/dataset/mnist'
train_file = 'mnist_train.csv'
test_file = 'mnist_test.csv'

model = Sequential()
output_dim = 10
total_train_sample = 60000
total_test_sample = 10000
image_dim = 784
model.add(layers.Dense(100, name='hidden_layer',
                       activation='relu', input_dim=image_dim))
model.add(layers.Dense(10, name='output_layer', activation='softmax'))

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer='sgd', metrics=['accuracy'])


def getInputOutput(fileName, samplecount):
    inputs = []
    targets = []
    counter = 0

    train_data = []
    start = time.time()
    with open(os.path.join(dir, fileName)) as f:
        train_data = f.readlines()
    print('read file cost ' + str(time.time() - start) + ' second')

    for line in train_data:
        line_array = line.split(',')

        targets.append(int(line_array[0]))
        inputs.append(np.asfarray(line_array[1:]))

        counter += 1
        if counter % 100 == 0:
            print('progress ' + str(counter * 100.0 / samplecount))

    targets = utils.to_categorical(np.array(targets), num_classes=10)
    return np.array(inputs), targets


def executeTrain():
    inputs, targets = getInputOutput(train_file, total_train_sample)
    model.fit(inputs, targets, verbose=1, epochs=4)


def executeTest():
    inputs, targets = getInputOutput(test_file, total_test_sample)
    lose, accuracy = model.evaluate(inputs, np.array(targets), verbose=1)
    print("\n\nAccuracy:" + str(accuracy * 100))


executeTrain()
executeTest()
