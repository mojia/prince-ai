import time


class PerformanceModel:

    def __init__(self, x_shape):
        self.create_on = time.localtime()
        self.x_shape = x_shape

    def setDataFields(self, codes, start_time, end_time,
                      back_window_length, future_window_length,
                      skipStep, min_size_samples):
        self.codes = codes
        self.start_time = start_time
        self.end_time = end_time
        self.back_window_length = back_window_length
        self.future_window_length = future_window_length
        self.skipStep = skipStep
        self.min_size_samples = min_size_samples

    def setModelFields(self, hidden_layer_1_unit, epochs, activation, optimizer, loss, k_fold):
        self.hidden_layer_1_unit = hidden_layer_1_unit
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.k_fold = k_fold

    def setFeatures(self, add_bar_features, add_dif_features, add_dea_features):
        self.add_bar_features = add_bar_features
        self.add_dif_features = add_dif_features
        self.add_dea_features = add_dea_features
