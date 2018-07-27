
import os
import numpy as np
import matplotlib.pyplot as plt
dir = '/Users/xinwang/ai/dataset/mnist'
fileName = 'mnist_train_100.csv'


def loadData(fileName):
    data = []
    with open(os.path.join(dir, fileName), 'r') as f:
        for line in f:
            array = line.strip().split(',')[1:]
            data.append(array)

    return np.asfarray(data),


def pca(data, topN=99999):
    print('data:' + str(data.shape))
    meanVals = np.mean(data, axis=0)

    meanRemoved = data - meanVals
    print('meanRemoved:' + str(meanRemoved.shape))

    cov = np.cov(meanRemoved, rowvar=0)
    print('cov' + str(cov.shape))

    eigVals, eigVects = np.linalg.eig(np.mat(cov))

    # ascending
    eigValIndex = np.argsort(eigVals)
    eigValIndex = eigValIndex[:-(topN + 1):-1]

    primaryVects = eigVects[:, eigValIndex]

    lowDData = meanRemoved * primaryVects
    reconMat = (lowDData * primaryVects.T) + meanVals
    return lowDData, reconMat


def showPCA():
    dataSet = loadData(fileName)
    lowMat, reconMat = pca(dataSet, 784)

    lowMat = lowMat.astype(int)
    reconMat = reconMat.astype(int)

    plt.imshow(lowMat)
    plt.show()
