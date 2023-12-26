from utility.values import *

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataLoaderWrapper:
    def __init__(self, dataMainPath):
        self.dataMainPath = dataMainPath

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

    @staticmethod
    def create_dataloaders(train_features, train_labels, test_features=None, test_labels=None, batch_size=32):
        train_dataset = TimeSeriesDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_loader = None
        if test_features is not None and test_labels is not None:
            test_dataset = TimeSeriesDataset(test_features, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    

if __name__ == '__main__':
    dataLoaderWrapper = DataLoaderWrapper(dataMainPath=dataMainPath)
    file_names = ["sample_data4learning.mat", "sample_data4learning2.mat"]
    X_train_val, Y_train_val, X_test, Y_test = dataLoaderWrapper.split_data(train_val_samples=20, test_samples=4)