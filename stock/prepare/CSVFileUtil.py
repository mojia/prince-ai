import csv
import os
import numpy as np

dir = '/Users/xinwang/ai/dataset/stock/train/'


class CSVFileUtil:

    def fileExist(fileName):
        if (os.path.exists(os.path.join(dir, fileName))):
            return True
        else:
            return False

    def readCSV(fileName):
        with open(os.path.join(dir, fileName)) as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            return np.array(data)

    def writeCSV(fileName, data):
        with open(os.path.join(dir, fileName), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
