import pandas as pd
import os
import importlib
import chardet
import sys
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

dir = '/Users/xinwang/ai/dataset/kaggle/predict-future-sales/predict-future-sales/'
shopsFile = 'shops.csv'
itemsFile = 'items.csv'
itemCategoriesFile = 'item_categories.csv'
salesFile = 'sales_train.csv'
itemCategoriesFile = 'item_categories.csv'


def howManyItems():
    with open(os.path.join(dir, itemsFile), 'r', encoding="utf-8") as f:
        for line in f:
            array = line.split(",")
            print(array[1])


def shopItemCnt():
    salesdf = pd.read_csv(os.path.join(dir, salesFile))
    # print(salesdf)
    sumGroupByDate = salesdf['item_cnt_day'].groupby([salesdf['item_id'], salesdf['shop_id'], salesdf['date']]).sum()
    print(str(sumGroupByDate))


shopItemCnt()
