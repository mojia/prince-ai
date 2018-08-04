import pymysql
from KLine import KLine
from config import config

defaultKtype = '5f'
futureWindow = config.futureWindow
fields = "id,  type, code, create_time, open_price, \
       close_price, high_price, low_price, volume, turnover, \
        label, dif, dea, bar "


class DBHelper:

    def __init__(self):
        self.db = pymysql.connect(host="localhost", port=3306, user="root",
                                  passwd="", db='stock', charset='utf8')

    def convert2KLineModel(self, dbRecord):
        k = KLine(dbRecord[0], dbRecord[1], dbRecord[2],
                  dbRecord[3], dbRecord[4],
                  dbRecord[5], dbRecord[6], dbRecord[7],
                  dbRecord[8], dbRecord[9])

        if len(dbRecord) == 10:
            k.setLabel(dbRecord[10])

        if len(dbRecord) == 14:
            k.setLabel(dbRecord[10])
            k.setMacdValues(dbRecord[11], dbRecord[12], dbRecord[13])

        return k

    def convert2KLineModels(self, data):
        klines = []
        for dbRecord in data:
            k = self.convert2KLineModel(dbRecord)
            klines.append(k)

        return klines

    def updateLabel(self, k):
        cursor = self.db.cursor()

        sql = 'update k_table set label= %(label)s where id=%(id)s'
        param = {
            "label": k.label,
            "id": k.id
        }

        try:
            cursor.execute(sql, param)
            self.db.commit()
        except Exception as e:
            print('updateLabel failed k:' + str(k) + e)
            self.db.rollback()

        cursor.close()

    def queryFutureKLines(self, code, start, type=defaultKtype):
        cursor = self.db.cursor()

        sql = "select " + fields + "from k_table \
               where code=%(code)s and create_time>= %(start)s \
               and type=%(type)s order by create_time \
               limit %(window)s"

        param = {
            "code": code,
            "start": start,
            "type": type,
            "window": futureWindow
        }
        cursor.execute(sql, param)
        data = cursor.fetchall()
        cursor.close()

        return self.convert2KLineModels(data)

    def query(self, code, startDateTime, endDateTime, ktype=defaultKtype):

        sql = "select " + fields + " from k_table \
            where type = '5f' and code = %(code)s \
            and create_time >= %(start)s \
            and create_time <= %(end)s \
            order by create_time"

        param = {
            "code": code,
            "start": startDateTime,
            "end": endDateTime
        }

        cursor = self.db.cursor()

        cursor.execute(sql, param)
        data = cursor.fetchall()

        cursor.close()

        return self.convert2KLineModels(data)

    def updateMACD(self, id, dif, dea, bar):
        cursor = self.db.cursor()

        sql = "update k_table \
        set dif = %(dif)s, dea = %(dea)s, bar = %(bar)s \
        where id = %(id)s"

        param = {
            "dif": dif,
            "dea": dea,
            "bar": bar,
            "id": id
        }

        try:
            cursor.execute(sql, param)
            self.db.commit()
        except Exception as e:
            print('updateMACD failed. ' + str(e))
            self.db.rollback()

        cursor.close()

    def queryKById(self, id):
        if id == "":
            return

        cursor = self.db.cursor()
        sql = "select id,type,create_time,open_price,close_price, \
        high_price,low_price,volume,turnover from k_table where id=" + id
        cursor.execute(sql)
        data = cursor.fetchone()
        cursor.close()

        return data

    def exist(self, k):
        data = self.query(k.code, k.createTime, k.createTime)
        if data is not None and len(data) > 0:
            return True
        else:
            return False

    def insertNewK(self, k):
        cursor = self.db.cursor()

        sql = "insert into k_table (type,code,create_time,open_price, \
        close_price,high_price,low_price,volume,turnover) \
        values (%(type)s,%(code)s,%(createTime)s,%(openPrice)s, \
        %(closePrice)s,%(highPrice)s,%(lowPrice)s,%(volume)s,%(turnover)s)"

        kline = {
            'type': k.type,
            'code': k.code,
            'createTime': k.createTime,
            'openPrice': k.openPrice,
            'closePrice': k.closePrice,
            'highPrice': k.highPrice,
            'lowPrice': k.lowPrice,
            'volume': k.volume,
            'turnover': k.turnover
        }
        try:
            cursor.execute(sql, kline)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print('save k exception, rollback. ' + str(e))

        cursor.close()

    def saveKLine(self, k):
        if k:
            if self.exist(k) is True:
                pass
            else:
                self.insertNewK(k)

    def __del__(self):
        self.db.close()
