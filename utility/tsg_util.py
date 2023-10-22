mainPath = "C:/Users/inho/OneDrive - 연세대학교 (Yonsei University)/INHO_/2_UMich_MS/2023 Fall/Kolmanovsky Lab/TimeShiftGoverner_DL/"
modelMainPath = mainPath + "model/"
dataMainPath = mainPath + "data/"

import pandas as pd
import numpy as np
import scipy.io
import sys
import os
from time import time
from sklearn.preprocessing import MinMaxScaler


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def make_sequence_dataset(feature, label, windowsize):
    feature_list = []
    label_list = []
    for i in range(len(label) - windowsize):
        feature_list.append(feature[i: i + windowsize])
        label_list.append(label[i + windowsize])
    return np.array(feature_list), np.array(label_list)


def make_sliding_window(feature, label, windowsize):
    X_sequenced, Y_sequenced = [], []
    for i in range (len(feature[0,0,:])):
        X, Y = make_sequence_dataset(feature[:,:,i], label[:,i], windowsize)
        try:
            X_sequenced = np.concatenate((X_sequenced, X), axis = 0)
            Y_sequenced = np.concatenate((Y_sequenced, Y), axis = 0)
        except:
            X_sequenced = X
            Y_sequenced = Y
    return X_sequenced, Y_sequenced


def GenerateNormalization(TrainDataOutput, TestDataOutput):
    scaler = MinMaxScaler(feature_range=(0, 1))

    Y_TrainDataOutput, Y_TestDataOutput = [], []
    for i in range (len(TrainDataOutput[0,:])):
        try:
            Y_TrainDataOutput = np.concatenate((Y_TrainDataOutput, TrainDataOutput[:,i]), axis = 0)
        except:
            Y_TrainDataOutput = TrainDataOutput
    for i in range (len(TestDataOutput[0,:])):
        try:
            Y_TestDataOutput = np.concatenate((Y_TestDataOutput, TestDataOutput[:,i]), axis = 0)
        except:
            Y_TestDataOutput = TestDataOutput
    
    DataOutput = np.concatenate([Y_TrainDataOutput, Y_TestDataOutput], axis = 0)
    # Discard Imaginary Part from complex data
    real_data = np.real(DataOutput).reshape(-1,1)
    # Fit and transform each sensor array using the fit_transform method of the corresponding MinMaxScaler instance
    normalized_data = scaler.fit_transform(real_data)

    return scaler


def NormalizeAllOutputDataWithScaler(scaler, TrainDataOutput, TestDataOutput):
    Train_length = len(TrainDataOutput)
    data = np.real(TrainDataOutput).reshape(-1,1)
    NormalizedTrainDataOutput = scaler.transform(data)
    NormalizedTrainDataOutput = NormalizedTrainDataOutput.reshape(Train_length,-1)
    Test_length = len(TestDataOutput)
    data = np.real(TestDataOutput).reshape(-1,1)
    NormalizedTestDataOutput = scaler.transform(data)
    NormalizedTestDataOutput = NormalizedTestDataOutput.reshape(Test_length,-1)

    return NormalizedTrainDataOutput, NormalizedTestDataOutput


def DeNormalizeOutputData(scaler, pred_norm, Y_test):
    pred = pred_norm
    pred = scaler.inverse_transform(pred_norm.reshape(-1,1)).reshape(-1)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(-1)

    return pred, Y_test


'''
# Data pre-processing - 1 involved among 1
def df_to_X_y(df, window_size=5):
# [[[t1], [t2], [t3], [t4], [t5]]] -> predict [t6]
# [[[t2], [t3], [t4], [t5], [t6]]] -> predict  [t7]
# [[[t3], [t4], [t5], [t6], [t7]]] -> predict  [t8]
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
      row = [[a] for a in df_as_np[i:i+window_size]]
      X.append(row)
      label = df_as_np[i+window_size]
      y.append(label)
    return np.array(X), np.array(y)


# Data pre-processing - 1 involved among 5
def df_to_X_y2(df, window_size=6):
# [[[t1, ds1], [t2, ds2], [t3, ds3], [t4, ds4], [t5, ds5]]] -> predict [t6]
# [[[t2, ds2], [t3, ds3], [t4, ds4], [t5, ds5], [t6, ds6]]] -> predict  [t7]
# [[[t3, ds3], [t4, ds4], [t5, ds5], [t6, ds6], [t7, ds7]]] -> predict  [t8]
    df_as_np = df.to_numpy()
    X = [] # n_train x n_timesteps x n_data
    y = []
    for i in range(len(df_as_np)-window_size):
      row = [r for r in df_as_np[i:i+window_size]]
      X.append(row)
      label = df_as_np[i+window_size][0]
      y.append(label)
    return np.array(X), np.array(y)


# Data pre-processing - 2 involved among 6
def df_to_X_y3(df, window_size=7):
    df_as_np = df.to_numpy()
    X = [] # n_train x n_timesteps x n_data
    y = []
    for i in range(len(df_as_np)-window_size):
      row = [r for r in df_as_np[i:i+window_size]]
      X.append(row)
      label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
      y.append(label)
    return np.array(X), np.array(y)


# pre-processing: Standardization                         
def preprocess(X,X2_train):
    temp_training_mean = np.mean(X2_train[:, :, 0])
    temp_training_std = np.std(X2_train[:, :, 0])
    X[:, :, 0] = (X[:, :, 0] - temp_training_mean) / temp_training_std
    return X


# pre-processing: Standardization 2 involved for both input and output                        
def preprocess3(X, X3_train):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3 = np.std(X3_train[:, :, 0])
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3 = np.std(X3_train[:, :, 1])
    
    # X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
    # X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3    
    # return X


def preprocess_output3(y, X3_train):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3 = np.std(X3_train[:, :, 0])
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3 = np.std(X3_train[:, :, 1])
    
    y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
    y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
    return y


# post-processing: Standardization 2 involved for both input and output
def postprocess_temp(arr, X3_train):    
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3  = np.std(X3_train[:, :, 1])
    
    arr = (arr*temp_training_std3) + temp_training_mean3
    return arr
    
def postprocess_p(arr, X3_train):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3  = np.std(X3_train[:, :, 0])    
    
    arr = (arr*p_training_std3) + p_training_mean3
    return arr

'''


