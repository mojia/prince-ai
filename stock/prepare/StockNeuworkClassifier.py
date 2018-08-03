import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt
from config import config
from FeatureCenter import FeatureCenter

model = Sequential()
output_dim = 3
features_dim = config.featureDim
backWindow = config.backWindowLength
debugTrainData = config.debugForPrepareData


model.add(layers.Dense(100, name='hidden_layer_1',
                       activation='tanh', input_dim=features_dim))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(16, name='hidden_layer_2', activation='tanh'))
# model.add(layers.Dense(8, name='hidden_layer_3', activation='tanh'))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(output_dim, name='output_layer', activation='softmax'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss = 'categorical_crossentropy'
model.compile(loss=loss,
              optimizer='adam', metrics=['categorical_accuracy'])


def drawTrainData(x, label='', plot=True, color='red'):
    fig = plt.figure(figsize=(30, 8))
    if plot is True:
        plt.plot(range(len(x)), x, color=color)
    plt.scatter(range(len(x)), x)
    plt.ylabel('Label:' + str(label))


def trainAndTest(codes, startDateTime, endDateTime):
    train_x = np.zeros((0, features_dim), dtype='float')
    train_y = np.zeros((0, 3), dtype='int')
    evaluate_x = np.zeros((0, features_dim), dtype='float')
    evaluate_y = np.zeros((0, 3), dtype='int')

    for i in range(len(codes)):
        featureCenter = FeatureCenter(codes[i], startDateTime, endDateTime)

        tx, ty, ex, ey = featureCenter.getDataSet()
        train_x = np.vstack((train_x, tx))
        train_y = np.vstack((train_y, ty))
        evaluate_x = np.vstack((evaluate_x, ex))
        evaluate_y = np.vstack((evaluate_y, ey))

    print('train_x.shape:' + str(train_x.shape) +
          ' train_y.shape:' + str(train_y))
    print('evaluate_x.shape:' + str(evaluate_x.shape) +
          ' evaluate_y.shape:' + str(evaluate_y))

    if debugTrainData is True:
        for i in range(len(train_x)):
            # drawTrainData(train_x[i], train_y[i])
            pass
    else:
        print('start training ... ')
        # batch_size=8
        history = model.fit(train_x, train_y, verbose=1, epochs=config.epochs)
        plt.plot(history.history['categorical_accuracy'])

        lose, accuracy = model.evaluate(evaluate_x, evaluate_y, verbose=1)
        print("lose:" + str(lose) + "\nAccuracy:" + str(accuracy * 100))
        plt.show()


trainAndTest(["SH600000", "SH600003", "SH600004"],
             "2001-01-09 09:35:00", "2018-10-23 09:35:00")
plt.show()
