import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error as mse

import copy
# ----------------
# Data pre-processing - 1 involved among 1
# ----------------
# --------------------------------------------------- 
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
# --------------------------------------------------- 

# ----------------
# Data pre-processing - 1 involved among 5
# ----------------
# --------------------------------------------------- 
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
# --------------------------------------------------- 

# ----------------
# Data pre-processing - 2 involved among 6
# ----------------
# --------------------------------------------------- 
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
# --------------------------------------------------- 

# ----------------
# pre-processing: Standardization
# ----------------
# ---------------------------------------------------                            
def preprocess(X,X2_train0):
    temp_training_mean = np.mean(X2_train[:, :, 0])
    temp_training_std = np.std(X2_train[:, :, 0])
    X[:, :, 0] = (X[:, :, 0] - temp_training_mean) / temp_training_std
    return X
# --------------------------------------------------- 

# ----------------
# pre-processing: Standardization 2 involved for both input and output
# ----------------
# ---------------------------------------------------                            
def preprocess3(X, X3_train0):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3 = np.std(X3_train[:, :, 0])
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3 = np.std(X3_train[:, :, 1])
    
    # X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
    # X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3    
    # return X

def preprocess_output3(y, X3_train0):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3 = np.std(X3_train[:, :, 0])
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3 = np.std(X3_train[:, :, 1])
    
    y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
    y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
    return y
# ---------------------------------------------------                            

# ----------------
# post-processing: Standardization 2 involved for both input and output
# ----------------
# --------------------------------------------------- 
def postprocess_temp(arr, X3_train0):    
    temp_training_mean3 = np.mean(X3_train[:, :, 1])
    temp_training_std3  = np.std(X3_train[:, :, 1])
    
    arr = (arr*temp_training_std3) + temp_training_mean3
    return arr
    
def postprocess_p(arr, X3_train0):
    p_training_mean3 = np.mean(X3_train[:, :, 0])
    p_training_std3  = np.std(X3_train[:, :, 0])    
    
    arr = (arr*p_training_std3) + p_training_mean3
    return arr
# --------------------------------------------------- 

# ----------------
# Prediction 1d
# ----------------
# --------------------------------------------------- 
def plot_predictions1(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
    
    plt.figure()
    plt.plot(df['Predictions'][start:end], label='Predictions')
    plt.plot(df['Actuals'][start:end], label='Actuals')
    plt.legend(fontsize=15)
    plt.xlabel('Prediction Time + 58 min (sec)',fontsize=15)
    plt.ylabel('Time shift (min)',fontsize=15)

    return df, mse(y, predictions)
# --------------------------------------------------- 
 
# ----------------
# Prediction 2d
# ----------------
# --------------------------------------------------- 
def plot_predictions2(model, X, y, X_train0, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds =predictions[:, 0], predictions[:, 1]
    p_actuals, temp_actuals = y[:, 0], y[:, 1]
    df = pd.DataFrame(data={'Temperature Predictions':temp_preds, 
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions':p_preds, 
                            'Pressure Actuals':p_actuals
                            })
    plt.figure()
    plt.plot(df['Temperature Predictions'][start:end], label='Temperature Predictions')
    plt.plot(df['Temperature Actuals'][start:end], label='Temperature Actuals')  
    plt.plot(df['Pressure Predictions'][start:end], label='Pressure Predictions')
    plt.plot(df['Pressure Actuals'][start:end], label='Pressure Actuals')  
    plt.legend(fontsize=15)
    return df[start:end], mse(y, predictions)
# --------------------------------------------------- 

# ----------------
# Prediction 2d - post
# ----------------
# --------------------------------------------------- 
def plot_post_predictions2(model, X, y, X_train0, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds = postprocess_p(predictions[:, 0], X_train0), postprocess_temp(predictions[:, 1], X_train0)
    p_actuals, temp_actuals = postprocess_p(y[:, 0], X_train0), postprocess_temp(y[:, 1], X_train0)
    df = pd.DataFrame(data={'Temperature Predictions':temp_preds, 
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions':p_preds, 
                            'Pressure Actuals':p_actuals
                            })
    plt.figure()
    plt.plot(df['Temperature Predictions'][start:end], label='Temperature Predictions')
    plt.plot(df['Temperature Actuals'][start:end], label='Temperature Actuals')  
    plt.plot(df['Pressure Predictions'][start:end], label='Pressure Predictions')
    plt.plot(df['Pressure Actuals'][start:end], label='Pressure Actuals')  
    plt.legend(fontsize=15)
    return df[start:end], mse(y, predictions)
# --------------------------------------------------- 

# zip_path=tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
# csv_path='C:\\Users\\ys-th\\.keras\\datasets\\jena_climate_2009_2016.csv'
# df=pd.read_csv(csv_path)
# df=df[5::6] # take every hours

mat1_file_name="result_alt550_ecc0.74_incl62.8_RAAN0_argp280.mat"
mat2_file_name="result_alt600_ecc0.74_incl62.8_RAAN0_argp280.mat"
mat1_file=scipy.io.loadmat(mat1_file_name)
mat2_file=scipy.io.loadmat(mat2_file_name)
# print(type(mat_file))
# for i in mat_file:
#     print(i)

time1_Deputy_Chief=mat1_file['Traj'][:,:13]
tbackward1=mat1_file['Traj'][:,23]

# time1_Deputy_Chief=mat1_file['Traj'][:4000,:13]
# tbackward1=mat1_file['Traj'][:4000,23]
# time2_Deputy_Chief=mat2_file['Traj'][:4000,:13]
# tbackward2=mat2_file['Traj'][:4000,23]


df=pd.DataFrame({'time'      :time1_Deputy_Chief[:,0],
                  't_backward':tbackward1,
                  'xD'        :time1_Deputy_Chief[:,1],
                  'yD'        :time1_Deputy_Chief[:,2],
                  'zD'        :time1_Deputy_Chief[:,3],
                  'vxD'       :time1_Deputy_Chief[:,4],
                  'vyD'       :time1_Deputy_Chief[:,5],
                  'vzD'       :time1_Deputy_Chief[:,6],
                  'xC'        :time1_Deputy_Chief[:,7],
                  'yC'        :time1_Deputy_Chief[:,8],
                  'zC'        :time1_Deputy_Chief[:,9],
                  'vxC'       :time1_Deputy_Chief[:,10],
                  'vyC'       :time1_Deputy_Chief[:,11],
                  'vzC'       :time1_Deputy_Chief[:,12]
                  })
# df=pd.DataFrame({'time'      :np.concatenate((time1_Deputy_Chief[:,0].real,time2_Deputy_Chief[:,0].real),axis=0),
#                  't_backward':np.concatenate((tbackward1,tbackward2),axis=0),
#                  'xD'        :np.concatenate((time1_Deputy_Chief[:,1],time2_Deputy_Chief[:,0]),axis=0),
#                  'yD'        :np.concatenate((time1_Deputy_Chief[:,2],time2_Deputy_Chief[:,0]),axis=0),
#                  'zD'        :np.concatenate((time1_Deputy_Chief[:,3],time2_Deputy_Chief[:,0]),axis=0),
#                  'vxD'       :np.concatenate((time1_Deputy_Chief[:,4],time2_Deputy_Chief[:,0]),axis=0),
#                  'vyD'       :np.concatenate((time1_Deputy_Chief[:,5],time2_Deputy_Chief[:,0]),axis=0),
#                  'vzD'       :np.concatenate((time1_Deputy_Chief[:,6],time2_Deputy_Chief[:,0]),axis=0),
#                  'xC'        :np.concatenate((time1_Deputy_Chief[:,7],time2_Deputy_Chief[:,0]),axis=0),
#                  'yC'        :np.concatenate((time1_Deputy_Chief[:,8],time2_Deputy_Chief[:,0]),axis=0),
#                  'zC'        :np.concatenate((time1_Deputy_Chief[:,9],time2_Deputy_Chief[:,0]),axis=0),
#                  'vxC'       :np.concatenate((time1_Deputy_Chief[:,10],time2_Deputy_Chief[:,0]),axis=0),
#                  'vyC'       :np.concatenate((time1_Deputy_Chief[:,11],time2_Deputy_Chief[:,0]),axis=0),
#                  'vzC'       :np.concatenate((time1_Deputy_Chief[:,12],time2_Deputy_Chief[:,0]),axis=0),
#                  })

# plt.figure()
# df['t_backward'].plot()
# plt.xlabel('Time (min)',fontsize=15)
# plt.ylabel('Time shift (min)',fontsize=15)

df.index=pd.to_datetime(df['time'], unit='minute')
df=df.drop('time', axis=1)
df.head()




window_size=10
num_variables=13
X2, y2 = df_to_X_y2(df, window_size=window_size)
X2.shape, y2.shape

X2_train, y2_train = X2[:3000], y2[:3000]
X2_val, y2_val = X2[3000:3500], y2[3000:3500]
X2_test, y2_test = X2[3500:], y2[3500:]
X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape


# X2_train, y2_train = X2[:6000], y2[:6000]
# X2_val, y2_val = X2[6000:6500], y2[6000:6500]
# X2_test, y2_test = X2[6500:7000], y2[6500:7000]
# X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape
# X2_train0=copy.deepcopy(X2_train)
# # preprocess
# preprocess(X2_train,X2_train0)
# preprocess(X2_val,  X2_train0)
# preprocess(X2_test, X2_train0)

model_lstm1 = Sequential()
model_lstm1.add(InputLayer((window_size, num_variables)))
# model_lstm1.add(LSTM(32, return_sequences=True))
model_lstm1.add(LSTM(64))
model_lstm1.add(Dense(8, 'relu'))
model_lstm1.add(Dense(1, 'linear'))
model_lstm1.summary()

cp1 = ModelCheckpoint('model_lstm1/', save_best_only=True)
model_lstm1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model_lstm1.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), epochs=10, callbacks=[cp1])
plot_predictions1(model_lstm1, X2_test, y2_test)


# model_lstm1.add(Conv1D(64, kernel_size=2))
# model_lstm1.add(Flatten())




# df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

# temp = df['T (degC)']
# temp.plot()

# WINDOW_SIZE = 5 # window of 5 steps
# X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
# X1.shape, y1.shape

# X_train1, y_train1 = X1[:60000], y1[:60000]
# X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
# X_test1, y_test1 = X1[65000:], y1[65000:]
# X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

# # Design structure of LSTM 
# model1 = Sequential()
# model1.add(InputLayer((5, 1)))
# model1.add(LSTM(64))
# model1.add(Dense(8, 'relu'))
# model1.add(Dense(1, 'linear'))

# model1.summary()

# # Create an LSTM object
# cp1 = ModelCheckpoint('model1/', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# # Learning process using train and validation sets.
# model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])

# model1 = load_model('model1/')

# # ## Result of Training Data Set
# # train_predictions = model1.predict(X_train1).flatten()
# # train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
# # plt.plot(train_results['Train Predictions'][50:100])
# # plt.plot(train_results['Actuals'][50:100])
# # # plt.plot(train_results['Actuals'][50:100]-train_results['Train Predictions'][50:100])

# # ## Result of Validation Data Set
# # val_predictions = model1.predict(X_val1).flatten()
# # val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
# # plt.plot(val_results['Val Predictions'][:100])
# # plt.plot(val_results['Actuals'][:100])

# ## Result of TEST Data Set
# test_predictions = model1.predict(X_test1).flatten()
# test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
# plt.plot(test_results['Test Predictions'][:100])
# plt.plot(test_results['Actuals'][:100])

# plot_predictions1(model1, X_test1, y_test1)

# ## Practice Multivariate LSTM
# temp_df = pd.DataFrame({'Temperature':temp})
# temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
# # temp_df

# day = 60*60*24
# year = 365.2425*day

# temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
# temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
# temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
# temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
# temp_df.head()
# temp_df = temp_df.drop('Seconds', axis=1)
# temp_df.head()

# X2, y2 = df_to_X_y2(temp_df, window_size=6)
# X2.shape, y2.shape

# X2_train, y2_train = X2[:60000], y2[:60000]
# X2_val, y2_val = X2[60000:65000], y2[60000:65000]
# X2_test, y2_test = X2[65000:], y2[65000:]
# X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape
# X2_train0=copy.deepcopy(X2_train)
# # preprocess
# preprocess(X2_train,X2_train0)
# preprocess(X2_val,  X2_train0)
# preprocess(X2_test, X2_train0)

# model4 = Sequential()
# model4.add(InputLayer((6, 5)))
# model4.add(LSTM(64))
# model4.add(Dense(8, 'relu'))
# model4.add(Dense(1, 'linear'))
# model4.summary()

# cp4 = ModelCheckpoint('model4/', save_best_only=True)
# model4.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# model4.fit(X2_train, y2_train, validation_data=(X2_val, y2_val), epochs=10, callbacks=[cp4])
# plot_predictions1(model4, X2_test, y2_test)



# p_temp_df = pd.concat([df['p (mbar)'], temp_df], axis=1)
# p_temp_df.head()

# X3, y3 = df_to_X_y3(p_temp_df, window_size=7)
# X3.shape, y3.shape

# X3_train, y3_train = X3[:60000], y3[:60000]
# X3_val, y3_val = X3[60000:65000], y3[60000:65000]
# X3_test, y3_test = X3[65000:], y3[65000:]
# X3_train.shape, y3_train.shape, X3_val.shape, y3_val.shape, X3_test.shape, y3_test.shape
# X3_train0=copy.deepcopy(X3_train)

# preprocess3(X3_train, X3_train0)
# preprocess3(X3_val, X3_train0)
# preprocess3(X3_test, X3_train0)

# preprocess_output3(y3_train, X3_train0)
# preprocess_output3(y3_val, X3_train0)
# preprocess_output3(y3_test, X3_train0)


# model5 = Sequential()
# model5.add(InputLayer((7, 6)))
# model5.add(LSTM(32, return_sequences=True))
# model5.add(Conv1D(64, kernel_size=2, activation='relu'))
# model5.add(LSTM(64))
# model5.add(Dense(8, 'relu'))
# model5.add(Dense(2, 'linear'))
# model5.summary()

# cp5 = ModelCheckpoint('model5/', save_best_only=True)
# model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=20, callbacks=[cp5])

# plot_predictions2(model5, X3_test, y3_test, X3_train0, start=0, end=100)

# post_processed_df=plot_post_predictions2(model5, X3_test, y3_test, X3_train0, start=0, end=100)