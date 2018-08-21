from sklearn import svm
from DataCenter import DataCenter
from Config import Config


class StockSVM:
    def __init__(self, codes, startDateTime, endDateTime):
        self.codes = codes
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.dataCenter = DataCenter()
        self.config = Config()

    def show_accuracy(re1, re2):
        for i in range()

    def trySVM(X, Y, evaluate_x, evaluate_y):
        clf = svm.SVC(c=0.8, kernel='rbf', gamma='20', decision_function_shape='ovr')
        clf.fit(X, Y)

        predict_y = clf.predict(evaluate_x)
        show_accuracy(predict_y, evaluate_y)

    def trainAndTest(self):
        X, Y, evaluate_x, evaluate_y = self.loadData(self.codes, self.startDateTime, self.endDateTime)

        if self.config.debugForPrepareData is False:
            self.trySVM(X, Y, evaluate_x, evaluate_y)


if __name__ == '__main__':
    StockSVM(
        ["SH600000", "SH600004", "SH600005", "SH600006", "SH600007", "SH600008"], "2001-01-09 09:35:00", "2017-10-23 09:35:00").trainAndTest()
