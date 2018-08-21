import pymysql
from KLine import KLine
from Config import Config
config = Config()

fields = "codes,start_time,end_time,back_window_length,future_window_length,epochs, \
skipStep,min_size_samples,k_fold,hidden_layer_1_unit,activation,add_bar_features,\
add_dif_features,add_dea_features,x_shape,create_on,optimizer,loss"


class PerformanceDBHelper:
    def __init__(self):
        self.db = pymysql.connect(host="localhost", port=3306, user="root",
                                  passwd="", db='stock', charset='utf8')

    def buildInsertSQL(self):
        sql = "insert into performance_table( " + fields + " ) " + \
            "values( %(codes)s, %(start_time)s, %(end_time)s, %(back_window_length)s, \
        %(future_window_length)s, %(epochs)s, %(skipStep)s, %(min_size_samples)s, \
        %(k_fold)s,%(hidden_layer_1_unit)s, %(activation)s, %(add_bar_features)s, \
        %(add_dif_features)s, %(add_dea_features)s, %(x_shape)s, %(create_on)s, \
        %(optimizer)s, %(loss)s )"

        return sql

    def buildInsertParam(self, model):
        param = {
            "codes": model.codes,
            "start_time": model.start_time,
            "end_time": model.end_time,
            "back_window_length": model.back_window_length,
            "future_window_length": model.future_window_length,
            "epochs": model.epochs,
            "skipStep": model.skipStep,
            "min_size_samples": model.min_size_samples,
            "k_fold": model.k_fold,
            "hidden_layer_1_unit": model.hidden_layer_1_unit,
            "activation": model.activation,
            "add_bar_features": model.add_bar_features,
            "add_dif_features": model.add_dif_features,
            "add_dea_features": model.add_dea_features,
            "x_shape": model.x_shape,
            "create_on": model.create_on,
            "optimizer": model.optimizer,
            "loss": model.loss
        }

        return param

    def insertNewPerformanceModel(self, model):
        cursor = self.db.cursor()

        sql = self.buildInsertSQL()
        param = self.buildInsertParam()

        try:
            cursor.execute(sql, param)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print('insertNewPerformanceModel exception, rollback. ' + str(e))

        cursor.close()

    def updatePerformanceFields(self, id, train_accuracy, evaludate_accuracy):
        cursor = self.db.cursor()
        sql = "update performance_table " + \
            "set train_accuracy=%(train_accuracy)s, evaludate_accuracy=%(evaludate_accuracy)s " + \
            "where id=%(id)s"
        param = {
            "train_accuracy": train_accuracy,
            "evaludate_accuracy": evaludate_accuracy,
            "id": id
        }

        try:
            cursor.execute(sql, param)
            self.db.commit()
        except Exception as e:
            print('updatePerformanceFields failed. ' + str(e))
            self.db.rollback()

        cursor.close()
