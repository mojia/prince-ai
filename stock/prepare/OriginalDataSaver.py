from KLine import KLine
from DBHelper import DBHelper
from datetime import datetime

import os

dir = '/Users/xinwang/Downloads/Stk_Min5_QFQ_20180802/'
kfile = 'SH600004.CSV'

ktype5F = '5f'


class OriginalDataSaver:

    def __init__(self, csvFileName):
        self.csvFileName = csvFileName
        self.dbHelper = DBHelper()
        self.code = self.csvFileName.replace('.CSV', '')

    def taobaoShop(self, array):
        # for taobao数据商家
        createTime = array[0].strip() + ' ' + array[1].strip()
        createTime_object = datetime.strptime(createTime, '%Y/%m/%d %H:%M')

        k = KLine(None, ktype5F, self.code, createTime_object,
                  float(array[2].strip()), float(array[5].strip()),
                  float(array[3].strip()), float(array[4].strip()),
                  int(array[6].strip()), int(array[7].strip()))
        return k

    def tongxinda(self, array):
        # for 通信达
        createTime = array[0].strip() + ' ' + array[1].strip()
        createTime_object = datetime.strptime(createTime, '%Y/%m/%d %H%M')

        k = KLine(None, ktype5F, '600846',
                  createTime_object,
                  float(array[2].strip()), float(array[5].strip()),
                  float(array[3].strip()), float(array[4].strip()),
                  int(array[6].strip()),
                  int(array[7].strip()))
        return k

    def buildKLine(self, array):
        if len(array) > 7:
            return self.taobaoShop(array)

    def processFile(self):
        print('OriginalDataSaver start...')
        with open(os.path.join(dir, self.csvFileName), 'r') as f:
            i = 0
            for line in f:
                array = line.split(',')
                k = self.buildKLine(array)
                self.dbHelper.saveKLine(k)

                i += 1
                if i % 1000 == 0:
                    print('OriginalDataSaver.processFile ' + self.code +
                          ' progress line num ' + str(i))

        print('OriginalDataSaver end')
