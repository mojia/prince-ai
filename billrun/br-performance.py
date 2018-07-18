import csv
import os
import numpy as np
import matplotlib.pyplot as plt

dir = '/Users/xinwang/Downloads'
dotFile = 'Dotloop-1806.csv'
zgfile = 'ZG-1806.csv'


def pickdata(path):
    cost = []
    with open(path, 'r') as dotcsv:
        lines = csv.reader(dotcsv)

        for line in lines:
            if lines.line_num == 1:
                continue
            if line[8] == "" or float(line[8]) <= 0 or float(line[8]) > 3000:
                continue
            # if float(line[14]) * 0.001 > 4000:
            #     continue
            c = float(line[14]) * 0.001 / float(line[8])
            #
            if c > 300:
                continue

            if c > 10 and path != zgfile:
                print(path + str('####') + str(line))

            cost.append(c)
    return cost


dotcost = pickdata(os.path.join(dir, dotFile))
dotcost.sort()

zgcost = pickdata(os.path.join(dir, zgfile))
zgcost.sort()

fig = plt.figure()
plt.scatter(range(len(dotcost)), dotcost, color='red', s=[1] * len(dotcost))
plt.scatter(range(len(zgcost)), zgcost, color='black', s=[1] * len(zgcost))

plt.show()
