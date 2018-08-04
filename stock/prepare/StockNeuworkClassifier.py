import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt
from config import config
from FeatureCenter import FeatureCenter

output_dim = 3
features_dim = config.featureDim
backWindow = config.backWindowLength
debugTrainData = config.debugForPrepareData


def createModel():
    model = Sequential()

    model.add(layers.Dense(config.hidden_layer_1_unit,
                           name='hidden_layer_1',
                           activation=config.activation,
                           input_dim=features_dim - 1))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(100, name='hidden_layer_2', activation='tanh'))
    # model.add(layers.Dense(8, name='hidden_layer_3', activation='tanh'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim,
                           name='output_layer', activation='softmax'))

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


def countLabel(y):
    count0 = 0
    count1 = 0
    count2 = 0
    for i in range(len(y)):
        yVector = y[i]
        index = np.argmax(yVector)
        if index == 0:
            count0 += 1
        if index == 1:
            count1 += 1
        if index == 2:
            count2 += 1
    if config.debugForPrepareData is True:
        print('count0 ' + str(count0))
        print('count1 ' + str(count1))
        print('count2 ' + str(count2))

    return count0, count1, count2


def trainAndTest(codes, startDateTime, endDateTime):
    train_x = np.zeros((0, features_dim - 1), dtype='float')
    train_y = np.zeros((0, 3), dtype='int')
    evaluate_x = np.zeros((0, features_dim - 1), dtype='float')
    evaluate_y = np.zeros((0, 3), dtype='int')

    for i in range(len(codes)):
        featureCenter = FeatureCenter(codes[i], startDateTime, endDateTime)

        tx, ty, ex, ey = featureCenter.getDataSet()
        print(codes[i] + ' tx\n' + str(len(tx)))
        print(codes[i] + ' ty\n' + str(len(ty)))
        print(codes[i] + ' ex\n' + str(len(ex)))
        print(codes[i] + ' ey\n' + str(len(ey)))
        train_x = np.vstack((train_x, tx))
        train_y = np.vstack((train_y, ty))
        evaluate_x = np.vstack((evaluate_x, ex))
        evaluate_y = np.vstack((evaluate_y, ey))

    if debugTrainData is True:
        for i in range(len(train_x)):
            # drawTrainData(train_x[i], train_y[i])
            pass
    else:
        print('start training ... ')
        print('Final train_x.shape:' + str(train_x.shape))
        print('Final evaluate_x.shape:' + str(evaluate_x.shape))
        countLabel(train_y)

        model = createModel()
        # batch_size=8
        history = model.fit(train_x, train_y, verbose=1, epochs=config.epochs)
        plt.plot(history.history['categorical_accuracy'])

        lose, accuracy = model.evaluate(evaluate_x, evaluate_y, verbose=1)
        print("lose:" + str(lose) + "\nAccuracy:" + str(accuracy * 100))
        plt.show()


trainAndTest(["SH600000", "SH600004", "SH600005", "SH600006",
              "SH600007", "SH600008"],
             "2001-01-09 09:35:00", "2018-10-23 09:35:00")
plt.show()
