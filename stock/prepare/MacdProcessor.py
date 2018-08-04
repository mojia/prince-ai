from DBHelper import DBHelper


class MacdProcessor:
    def __init__(self, code, startDateTime, endDateTime):
        self.dbHelper = DBHelper()
        self.code = code
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime

    def calculateEMA_26(self, k, previous_ema_26):
        current_ema_26 = 0

        if previous_ema_26 == 0:
            current_ema_26 = k.closePrice
        else:
            current_ema_26 = previous_ema_26 * 25.0 / 27.0 \
                + k.closePrice * 2.0 / 27.0

        return current_ema_26

    def calculateEMA_12(self, k, previous_ema_12):
        current_ema_12 = 0
        if previous_ema_12 == 0:
            current_ema_12 = k.closePrice
        else:
            current_ema_12 = previous_ema_12 * 11.0 / 13.0 \
                + k.closePrice * 2.0 / 13.0

        return current_ema_12

    def calculateDIF(self, current_ema_12, current_ema_26, previous_ema_12):
        if previous_ema_12 == 0:
            return 0
        else:
            return current_ema_12 - current_ema_26

    def saveMACD(self, id, dif, dea, bar):
        self.dbHelper.updateMACD(id, dif, dea, bar)

    def refreshMACD(self):
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)
        print(len(kLines))

        previous_ema_12 = 0
        previous_ema_26 = 0
        dif = 0
        previous_dea = 0
        bar = 0

        i = 0
        for k in kLines:
            current_ema_12 = self.calculateEMA_12(k, previous_ema_12)
            current_ema_26 = self.calculateEMA_26(k, previous_ema_26)

            dif = self.calculateDIF(
                current_ema_12, current_ema_26, previous_ema_12)
            current_dea = previous_dea * 0.8 + dif * 0.2
            bar = 2.0 * (dif - current_dea)

            # print('EMA12:' + str(current_ema_12)
            # + ' EMA26:' + str(current_ema_26)
            # + ' current_dea:' + str(current_dea)
            # + ' bar:' + str(bar))
            self.saveMACD(k.id, dif, current_dea, bar)

            previous_ema_12 = current_ema_12
            previous_ema_26 = current_ema_26
            previous_dea = current_dea

            if i % 1000 == 0:
                print(self.code + ' refreshMACD progress ' +
                      str(i * 100.0 / len(kLines)))
            i += 1


# refreshMACD('600846', "2001-10-01 09:35:00", "2020-10-20 09:35:00")
# MacdProcessor('SH600004', "2001-10-01 09:35:00",
# "2018-10-09 09:40:00").refreshMACD()
