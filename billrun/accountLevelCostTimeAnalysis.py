import os
import matplotlib.pyplot as plt
import math
import numpy as np

dir = '/Users/xinwang/Downloads'
dotFile = 'account-level-time-dot.csv'
zillFile = 'account-level-time-zill.csv'


def getCost(fileName):
    c = []
    with open(os.path.join(dir, fileName), 'r') as f:
        for line in f:
            line = line.replace('"', '')
            cost = float(line)

            c.append(cost)

    return c


def getData(data):
    count = 20
    result = [0] * count
    for i in range(count):
        basket = i * 10
        basketNext = (i + 1) * 10
        for k in range(len(data)):
            if data[k] >= basket and data[k] < basketNext:
                result[i] += 1
        i += 1

    return result


def drawHist():
    dotC = getCost(dotFile)
    zillC = getCost(zillFile)

    fig = plt.figure()
    dotCData = getData(dotC)
    zillCData = getData(zillC)

    plt.bar(range(len(dotCData)), dotCData, alpha=0.4, label='Dot', color='red')
    plt.bar(range(len(zillCData)), zillCData, alpha=0.4, label='Zill', color='black')
    # plt.hist(range(len(zillC)), bins=20, alpha=0.4, label='Zill', color='black')
    plt.show()


drawHist()
