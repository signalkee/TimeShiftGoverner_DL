import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Embedding, Dense, LSTM, GRU, Bidirectional, Dropout, Conv1D, MaxPooling1D, Attention, Flatten# Normalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Input, Model
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention as MHeadAttention
import keras_tuner as kt  # https://webnautes.tistory.com/1642


def lstm_model(window_size, num_variables, X_train, num_outputs):
    model = Sequential()
    norm_layer = Normalization(input_shape=(window_size, num_variables))
    norm_layer.adapt(X_train)
    model.add(norm_layer)
    model.add(LSTM(60, return_sequences=False, activation="tanh", input_shape=(window_size, num_variables),))
    model.add(Dropout(0.25))
    model.add(Dense(num_outputs, activation="tanh", kernel_regularizer=L1L2(l1=0, l2=0.002)))
    opt = Adam(learning_rate=0.00005)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    return model


def bi_lstm_model(window_size, num_variables, X_train, num_outputs):
    model = Sequential()
    norm_layer = Normalization(input_shape=(window_size, num_variables))
    norm_layer.adapt(X_train)
    model.add(norm_layer)
    model.add(
        Bidirectional(
            LSTM(160, return_sequences=True), input_shape=(window_size, num_variables),
        )
    )
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(6)))
    model.add(Dense(num_outputs, activation="tanh", kernel_regularizer=L1L2(l1=0, l2=0.00081)))
    opt = Adam(learning_rate=0.00008)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model


def MHA_bi_lstm_model(window_size, num_variables, X_train, num_outputs): 
    model = Sequential()
    norm_layer = Normalization(input_shape=(window_size, num_variables))
    norm_layer.adapt(X_train)
    model.add(norm_layer)
    model.add(Conv1D(96, 3, activation='relu'))
    model.add(MaxPooling1D(2)) 
    model.add(Bidirectional(LSTM(112, return_sequences=True), input_shape=(window_size, num_variables),))
    model.add(MHeadAttention(head_num=4))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(36)))
    model.add(Dense(num_outputs, activation="tanh", kernel_regularizer=L1L2(l1=0, l2=0.00043)))
    opt = Adam(learning_rate=0.0000926)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    return model


class MyHyperModel(kt.HyperModel):
    def __init__ (self, window_size, X_train):
        self.window_size = window_size
        self.X_train = X_train
        
    def build(self, hp):
        window_Size = self.window_size
        X_train = self.X_train
        hp_units1 = hp.Int("units_1", min_value=64, max_value=160, step=8)
        hp_units2 = hp.Int("units_2", min_value=32, max_value=160, step=8)
        hp_units3 = hp.Int("units_3", min_value=4, max_value=8, step = 4)
        hp_units4 = hp.Float("dropout_1", min_value=0.05, max_value=0.4, step=0.05)
        hp_units5 = hp.Int("units_4", min_value=4, max_value=60, step = 8)
        hp_regularizer = hp.Float("regularizer", min_value=0.0001, max_value=0.001)
        hp_learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-4, sampling="LOG")
        
        model = Sequential()
        norm_layer = Normalization(input_shape=(window_Size, 12))
        norm_layer.adapt(X_train)
        model.add(norm_layer)
        regularizer = L1L2(l1=0, l2=hp_regularizer)
        model.add(Conv1D(hp_units1, 3, activation='relu'))
        model.add(MaxPooling1D(2)) 
        model.add(Bidirectional(LSTM(hp_units2, return_sequences=True), input_shape=(80, 6),))
        model.add(MHeadAttention(head_num=hp_units3))
        model.add(Dropout(hp_units4))
        model.add(Bidirectional(LSTM(hp_units5)))


        # hp_units1 = hp.Int("units_1", min_value=64, max_value=160, step=8)
        # hp_units4 = hp.Float("dropout_1", min_value=0.0, max_value=0.3, step=0.05)
        # hp_regularizer = hp.Float("regularizer", min_value=0.0001, max_value=0.001)
        # hp_learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-4, sampling="LOG")
        # model = Sequential()
        # norm_layer = Normalization(input_shape=(window_Size, 6))
        # norm_layer.adapt(X_train)
        # model.add(norm_layer)
        # regularizer = L1L2(l1=0, l2=hp_regularizer)
        # model.add(LSTM(hp_units1, return_sequences=False, activation="tanh"))
        # model.add(Dropout(hp_units4))


        model.add(Dense(1, activation="tanh", kernel_regularizer=regularizer))
        model.summary()
        model.compile(
            loss="mean_squared_error", optimizer=Adam(learning_rate=hp_learning_rate),
            metrics=["accuracy"],)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", min_value=256,
                              max_value=512, step=256),
            **kwargs,
        )
