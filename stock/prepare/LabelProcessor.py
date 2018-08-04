from DBHelper import DBHelper
from KLine import KLine
from Utils import Utils
from KPeriod import KPeriod
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib.pylab import date2num
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import multiprocessing
from config import config


increase_decrease = "/\\"
decrease_increase = "\/"
rise = "///"
fall = "\\\\"

Action_B = 'BB'
Action_H = 'HH'
Action_S = 'SS'

# 波动区间
width = 5.0
interval_approx_width = 2.0


class LabelProcessor:
    def __init__(self, code, startDateTime, endDateTime):
        self.code = code
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime

        self.dbHelper = DBHelper()
        self.utils = Utils()
        self.debug = config.debugForPrepareData

    def getPeakPeriod(self, klines):
        closePriceList = []
        for k in klines:
            closePrice = k.closePrice
            closePriceList.append(closePrice)

        maxIndex = closePriceList.index(np.max(closePriceList))

        if maxIndex == 0:
            return KPeriod(klines[0], None, [klines[0], klines[1], klines[2]])
        elif maxIndex == len(klines) - 1:
            return KPeriod(klines[-1], None, [klines[-1],
                                              klines[-2], klines[-3]])
        else:
            return KPeriod(klines[maxIndex], None,
                           [klines[maxIndex - 1], klines[maxIndex],
                            klines[maxIndex + 1]])

    def getBottomPeriod(self, klines):
        closePriceList = []
        for k in klines:
            closePrice = k.closePrice
            closePriceList.append(closePrice)

        minIndex = closePriceList.index(np.min(closePriceList))

        if minIndex == 0:
            return KPeriod(None, klines[0], [klines[0], klines[1], klines[2]])
        elif minIndex == len(klines) - 1:
            return KPeriod(None, klines[-1], [klines[-1],
                                              klines[-2], klines[-3]])
        else:
            return KPeriod(None, klines[minIndex],
                           [klines[minIndex - 1], klines[minIndex],
                            klines[minIndex + 1]])

    def checkMode(self, k, peak, bottom):
        if (peak.getPeakTime() >= k.createTime) \
                and (peak.getPeakTime() < bottom.getBottomTime()):
            return increase_decrease
        elif (bottom.getBottomTime() >= k.createTime) \
                and (bottom.getBottomTime() < peak.getPeakTime()):
            return decrease_increase

        return 'unknown'

    def drawLines(self, futurekLines, mode, action, delterCH, delterCL):
        prices = []
        for k in futurekLines:
            prices.append(k.closePrice)

        fig = plt.figure(figsize=(30, 8))
        x = range(len(futurekLines))
        y = prices
        plt.plot(x, prices, c='r')
        plt.scatter(x, prices, c='r')

        label = mode + '\n' + action + ' CH:' + \
            str('%.1f' % delterCH) + ' CL:' + str('%.1f' % delterCL)
        plt.xlabel(label)

        for x, y in zip(x, y):
            if x % 5 == 0:
                plt.annotate('(%s)' % (futurekLines[x].closePrice),
                             xy=(x, y),
                             xytext=(0, -10),
                             textcoords='offset points',
                             ha='center',
                             va='top')

    def delter(self, k, period):
        return abs(k.closePrice - period.getPrice()) * 100.0 / k.closePrice

    def increaseDecreaseAction(self, delterCH):
        if delterCH > width:
            return Action_B
        elif delterCH < interval_approx_width:
            return Action_S
        else:
            return Action_H

    def decreaseIncreaseAction(self, delterCL):
        if delterCL > width:
            return Action_S
        if delterCL < interval_approx_width:
            return Action_B
        else:
            return Action_H

    def riseAction(delterCH):
        if delterCH > width:
            return Action_B
        else:
            return Action_H

    def fallAction(delterCL):
        if delterCL > width:
            return Action_S
        else:
            return Action_H

    def calculateAction(self, mode, k, delterCH, delterCL):
        if mode == increase_decrease:
            return self.increaseDecreaseAction(delterCH)
        elif mode == decrease_increase:
            return self.decreaseIncreaseAction(delterCL)
        else:
            return Action_H

    def drawKLinesOfColorActions(self):
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)

        for i in range(config.debugSampleCount):
            data = kLines[
                i * config.futureWindow: (i + 1) * config.futureWindow]
            prices = []
            color = []
            for k in data:
                print(str(k) + '\n')
                prices.append(k.closePrice)
                if k.label == 0:
                    color.append('red')
                elif k.label == 1:
                    color.append('blue')
                else:
                    color.append('green')
            print('color\n' + str(color))

            fig = plt.figure(figsize=(30, 8))
            x = range(len(data))
            y = prices
            plt.plot(x, prices, c='y')
            plt.scatter(x, prices, c=color)

            for x, y in zip(x, y):
                if x % 20 == 0:
                    plt.annotate('(%s)' % (data[x].closePrice),
                                 xy=(x, y),
                                 xytext=(0, -10),
                                 textcoords='offset points',
                                 ha='center',
                                 va='top')
        plt.show()

    def CL_CH_Analysis(self):
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)
        kLines = kLines[:10000]
        ch = []
        cl = []
        for i in range(len(kLines)):
            k = kLines[i]
            futurekLines = self.dbHelper.queryFutureKLines(
                k.code, k.createTime)
            cur = k.closePrice
            peakPeriod = self.getPeakPeriod(futurekLines)
            bottomPeriod = self.getBottomPeriod(futurekLines)

            delterCH = self.delter(k, peakPeriod)
            delterCL = self.delter(k, bottomPeriod)
            ch.append(delterCH)
            cl.append(delterCL)
            if i % 100 == 0:
                print(str(i) + ' progress ' + str(i * 100.0 / len(kLines)))
        basket = [0] * 11
        for i in range(len(ch)):
            if ch[i] < 5:
                basket[0] += 1
            elif ch[i] < 10:
                basket[1] += 1
            elif ch[i] < 15:
                basket[2] += 1
            elif ch[i] < 20:
                basket[3] += 1
            elif ch[i] < 25:
                basket[4] += 1
            elif ch[i] < 30:
                basket[5] += 1
            elif ch[i] < 35:
                basket[6] += 1
            elif ch[i] < 40:
                basket[7] += 1
            elif ch[i] < 45:
                basket[8] += 1
            elif ch[i] < 50:
                basket[9] += 1
            else:
                basket[10] += 1

        plt.hist(range(len(ch)), ch)
        # plt.scatter(range(len(basket)), basket, s=30, c='red')
        # plt.plot(range(len(cl)), cl, c='green', s=[1] * len(cl))
        # plt.scatter(range(len(cl)), cl, c='green', s=[1] * len(cl))
        print('CH mean ' + str(np.array(ch).mean()))
        print('CL mean ' + str(np.array(cl).mean()))
        plt.show()

    def calculateLabel(self, k, debug):
        futurekLines = self.dbHelper.queryFutureKLines(k.code, k.createTime)
        if len(futurekLines) < config.futureWindow:
            return 1

        closePrice = k.closePrice

        cur = closePrice
        peakPeriod = self.getPeakPeriod(futurekLines)
        bottomPeriod = self.getBottomPeriod(futurekLines)

        mode = self.checkMode(k, peakPeriod, bottomPeriod)

        delterCH = self.delter(k, peakPeriod)
        delterCL = self.delter(k, bottomPeriod)

        action = self.calculateAction(mode, k, delterCH, delterCL)

        if debug is True:
            self.drawLines(futurekLines, mode, action, delterCH, delterCL)

        if action == Action_B:
            return 0
        elif action == Action_H:
            return 1
        else:
            return 2

    def processOneKline(self, k):
        label = self.calculateLabel(k, self.debug)

        k.label = label
        self.dbHelper.updateLabel(k)

    def refreshLabels(self):
        print('LabelProcessor refreshLabels start...')
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)

        debugSample = 0
        for i in range(len(kLines)):
            if self.debug is False:
                self.processOneKline(kLines[i])
                if i % 1000 == 0:
                    print(self.code + ' LabelProcessor refreshLabels progress'
                          + str(i * 100.0 / len(kLines)))
            else:
                if i % 8 == 0:
                    self.processOneKline(kLines[i])
                    debugSample += 1
                    if debugSample == config.debugSampleCount:
                        break

        print('LabelProcessor refreshLabels end')


startDateTime = "2001-10-01 09:35:00"
endDateTime = "2020-10-20 09:35:00"
# codes = ['SH600000']
# codes = ['SH600001', 'SH600003', 'SH600004', 'SH600006']
codes = []


def updateLabels():
    for i in range(len(codes)):
        LabelProcessor(codes[i], startDateTime, endDateTime).refreshLabels()


# updateLabels()


# for debug
# LabelProcessor(codes[0], startDateTime, endDateTime)
# .drawKLinesOfColorActions()
# LabelProcessor(codes[0], startDateTime, endDateTime).CL_CH_Analysis()
