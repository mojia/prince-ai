import os
import matplotlib.pyplot as plt

dir = '/Users/xinwang/Downloads'
dotFile = 'account-level-time-dot.csv'
zillFile = 'account-level-time-zill.csv'


def getCost(fileName):
    c = []
    with open(os.path.join(dir, fileName), 'r') as f:
        for line in f:
            line = line.replace('"', '')
            cost = float(line) / 1000.0

            if cost < 10:
                c.append(cost)

    return c


dotC = getCost(dotFile)
zillC = getCost(zillFile)
dotC.sort()
zillC.sort()


def getBins():
    step = 0.25 / 1000.0

    bins = []
    for i in range(1000):
        value = step * i
        bins.append(value)

    return bins


def drawHist():
    fig = plt.figure()
    plt.hist(dotC, bins=getBins(), alpha=0.4,
             label='Dot', color='red', normed=1)
    plt.hist(zillC, bins=getBins(), alpha=0.4,
             label='Zill', color='black', normed=1)
    plt.show()


def drawLines():
    fig = plt.figure()
    plt.scatter(range(len(dotC)), dotC, color='red',
                s=[1] * len(dotC), label='Dot')
    plt.scatter(range(len(zillC)), zillC, color='black',
                s=[1] * len(zillC), label='Dot')

    plt.show()


drawLines()
