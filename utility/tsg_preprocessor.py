import pandas as pd
import numpy as np
import scipy.io
import sys
import os
from time import time
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesPreprocessor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.scaler = None

    def generate_normalization(self, train_data_output, test_data_output):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        y_train_data_output = np.real(train_data_output).reshape(-1, 1)
        y_test_data_output = np.real(test_data_output).reshape(-1, 1)
        data_output = np.concatenate([y_train_data_output, y_test_data_output], axis=0)

        self.scaler.fit_transform(data_output)
        return self.scaler

    def normalize_all_output_data_with_scaler(self, train_data_output, test_data_output):
        train_length = len(train_data_output)
        train_data = np.real(train_data_output).reshape(-1, 1)
        normalized_train_data_output = self.scaler.transform(train_data)
        normalized_train_data_output = normalized_train_data_output.reshape(train_length, -1)

        test_length = len(test_data_output)
        test_data = np.real(test_data_output).reshape(-1, 1)
        normalized_test_data_output = self.scaler.transform(test_data)
        normalized_test_data_output = normalized_test_data_output.reshape(test_length, -1)

        return normalized_train_data_output, normalized_test_data_output

    def make_sequence_dataset(self, feature, label):
        feature_list, label_list = [], []
        for i in range(len(label) - self.window_size):
            feature_list.append(feature[i: i + self.window_size])
            label_list.append(label[i + self.window_size])
        return np.array(feature_list), np.array(label_list)

    def make_sliding_window(self, feature, label):
        x_sequenced, y_sequenced = [], []
        for i in range(len(feature[0, 0, :])):
            x, y = self.make_sequence_dataset(feature[:, :, i], label[:, i])
            try:
                x_sequenced = np.concatenate((x_sequenced, x), axis=0)
                y_sequenced = np.concatenate((y_sequenced, y), axis=0)
            except:
                x_sequenced = x
                y_sequenced = y
        return x_sequenced, y_sequenced

    def denormalize_output_data(self, scaler, pred_norm, Y_test):
        pred_norm = np.array(pred_norm).reshape(-1, 1)
        pred = scaler.inverse_transform(pred_norm).flatten()
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        return pred, Y_test



if __name__ == '__main__':
    window_size = 10
    preprocessor = TimeSeriesPreprocessor(window_size)