from utility.values import *

import os
import numpy as np
import scipy.io

class DataLoader:
    def __init__(self, dataMainPath):
        self.dataMainPath = dataMainPath
        self.mat_files = []

    def load_mat_files(self, file_names):
        self.mat_files = [scipy.io.loadmat(os.path.join(self.dataMainPath, file)) for file in file_names]

    def split_data(self, train_val_samples=20, test_samples=4):
        X_train_val = []
        Y_train_val = []
        X_test = []
        Y_test = []

        for mat_file in self.mat_files:
            concatenated_data = mat_file['Traj_samples']
            time_Deputy_Chief = np.real(concatenated_data[:, 1:13])
            tbackward = np.real(concatenated_data[:, 23])

            # Split data into training/validation and test parts
            X_train_val.append(time_Deputy_Chief[:, :, :train_val_samples])
            Y_train_val.append(tbackward[:, :train_val_samples])
            X_test.append(time_Deputy_Chief[:, :, train_val_samples:train_val_samples + test_samples])
            Y_test.append(tbackward[:, train_val_samples:train_val_samples + test_samples])

        # Concatenate data from different files
        X_train_val = np.concatenate(X_train_val, axis=2)
        Y_train_val = np.concatenate(Y_train_val, axis=1)
        X_test = np.concatenate(X_test, axis=2)
        Y_test = np.concatenate(Y_test, axis=1)

        return X_train_val, Y_train_val, X_test, Y_test
    
    
if __name__ == '__main__':
    dataLoader = DataLoader(dataMainPath=dataMainPath)
    file_names = ["sample_data4learning.mat", "sample_data4learning2.mat"]
    dataLoader.load_mat_files(file_names)
    X_train_val, Y_train_val, X_test, Y_test = dataLoader.split_data(train_val_samples=20, test_samples=4)