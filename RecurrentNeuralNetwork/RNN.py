#!/usr/bin/python3

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

from datetime import datetime

from scipy.stats import rankdata

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/laptop/Documents/data/aviation/NASA/Challenge_Data/"

class RNN:

  def __init__(self, batch, epochs, inputDim, window):

    self.__batch = batch
    self.__epochs = epochs
    self.__input_dim = inputDim
    self.__window = window
    
    self.__model = None

  def __del__(self):

    del self.__batch
    del self.__epochs
    del self.__input_dim
    del self.__window
    del self.__model

  def build_model_1(self, layers):
    self.__model = Sequential()

    self.__model.add(LSTM(input_shape=(layers[1], layers[0]),
                   output_dim=layers[1], return_sequences=True))
                                          
    self.__model.add(Dropout(0.2))
    self.__model.add(LSTM(layers[2], return_sequences=True))
    self.__model.add(Dropout(0.2))

    self.__model.add(LSTM(layers[3], return_sequences=False))
    self.__model.add(Dropout(0.2))
                                      
    self.__model.add(Dense(output_dim=layers[3]))
    self.__model.add(Activation("linear"))

  def build_model(self):

    self.__model = Sequential()
    #self.__model.add(Dense(16, input_dim=self.__input_dim, activation='sigmoid'))
    #model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    #self.__model.add(LSTM(4, input_shape=(50, 3, 1), return_sequences=True))
    #self.__model.add(LSTM(units=4, input_shape=(1, 1), stateful=True))
    self.__model.add(LSTM(units=16, input_shape=(self.__window, self.__input_dim)))
    self.__model.add(LSTM(units=16))
    #self.__model.add(LSTM(16, return_sequences=True))
    #self.__model.add(Dropout(0.2))
    #self.__model.add(LSTM(units=16))
    #model.add(TimeDistributed(Dense(vocabulary)))
    #self.__model.add(Activation('linear'))
    self.__model.add(Dense(1, activation='linear'))

  def load_nasa_challenge_data(self, train_file, test_file, train_gap=20):

    columns = ['device_id', 'cycles', 'setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 23)]
    columns += sensors

    df_train = pd.read_csv(train_file, sep=' ', header=None)
    df_train.columns = columns
    df_train.dropna(axis=1, inplace=True)
    df_train['rank'] = df_train.groupby('device_id')['cycles'].rank(ascending=False)
    df_train['Y'] = np.zeros(df_train.shape[0])
    df_train.loc[df_train['rank'] <= train_gap, ['Y']] = 1
    df_train['RUL'] = np.zeros(df_train.shape[0])

    df_test = pd.read_csv(test_file, sep=' ', header=None)
    df_test.columns = columns
    df_test.dropna(axis=1, inplace=True)
    df_test['rank'] = df_test.groupby('device_id')['cycles'].rank(ascending=False)
    df_test['Y'] = np.zeros(df_test.shape[0])
    df_test.loc[df_test['rank'] <= train_gap, ['Y']] = 1
    df_test['RUL'] = np.zeros(df_test.shape[0])

    for dev_id in df_train['device_id'].unique():
      df_train.loc[df_train['device_id'] == dev_id, ['RUL']] = \
            df_train['cycles'].max() - df_train['cycles']

    for dev_id in df_test['device_id'].unique():
      df_test.loc[df_test['device_id'] == dev_id, ['RUL']] = \
            df_test['cycles'].max() - df_test['cycles']

    parameters = ['cycles', 'setting1', 'setting2', 'setting3']
    #parameters = ['setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 21)]
    parameters += sensors

    df_train.sort_values(by=['cycles'], ascending=True, inplace=True)
    df_test.sort_values(by=['cycles'], ascending=True, inplace=True)

    d = {}
    d['device_id'] = []
    d['cycles'] = []
    for dev_id in df_train['device_id'].unique():
      d['device_id'].append(dev_id)
      d['cycles'].append(df_train[df_train['device_id'] == dev_id]['cycles'].max())

    df_train_events = pd.DataFrame(data=d)
    df_train_events.sort_values(by=['cycles'], ascending=True, inplace=True)

    d = {}
    d['device_id'] = []
    d['cycles'] = []
    for dev_id in df_test['device_id'].unique():
      d['device_id'].append(dev_id)
      d['cycles'].append(df_test[df_test['device_id'] == dev_id]['cycles'].max())

    df_test_events = pd.DataFrame(data=d)
    df_test_events.sort_values(by=['cycles'], ascending=True, inplace=True)

    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(df_train[parameters].values)
    y_train = np.asarray(df_train['RUL']).ravel()
    X_test = scaler.fit_transform(df_test[parameters].values)
    y_test = np.asarray(df_test['RUL']).ravel()

    n = X_train.shape[0]
    
    X_train_seq = []
    y_train_seq = []
    for k in range(n - self.__window + 1):
      X_train_seq.append(X_train[k : k + self.__window])
      y_train_seq.append(y_train[k : k + self.__window])

    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)

    n = X_test.shape[0]
    X_test_seq = []
    y_test_seq = []
    for k in range(n - self.__window + 1):
      X_test_seq.append(X_test[k : k + self.__window])
      y_test_seq.append(y_test[k : k + self.__window])

    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)

    print(X_train_seq.shape)
    #X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], 1, X_train_seq.shape[1]))
    #X_test_seq = np.reshape(X_test_seq, (X_test_seq.shape[0], 1, X_test_seq.shape[1]))

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, df_train_events, df_test_events

  def fit(self, X, y, X_test, y_test, events):

    #self.build_model([1, sequence_length, int(sequence_length*2), int(sequence_length*4), 1])
    self.build_model_1([1, self.__window, int(self.__window*2), int(self.__window*4), 1])
    self.__model.summary()

    #self.__model.compile(optimizer='sgd', loss='mean_absolute_percentage_error', metrics=['mae', 'acc'])
    self.__model.compile(optimizer='rmsprop', loss='mean_absolute_percentage_error', metrics=['mae', 'acc'])
    #self.__model.compile(optimizer='sgd', loss=longitudinal_loss(events), metrics=['categorical_accuracy'])
    history = self.__model.fit(X, y, \
                    batch_size=self.__batch, epochs=self.__epochs, \
                    verbose=1, validation_data=(X_test, y_test))
    score = self.__model.evaluate(X_test, y_test, verbose=1)
    print(score)

    #metric = Metrics()
    #y_hat_test = self.__model.predict(X_test)
    y_hat_train = self.__model.predict(X)
    y_hat_test = self.__model.predict(X_test)

    #y_hat_test = y_hat_test[:,0]
    #print(confusion_matrix(y_test, y_hat_test))
    print(y_hat_train)
    print(y_hat_train.mean())
    print(y_hat_train.shape)

    print(y_hat_test)
    print(y_hat_test.mean())
    print(y_hat_test.shape)


class TestRNN(TestCase):

  def setUp(self):

    self.__rnn = RNN(20, 10, 25, 1)
    
  def tearDown(self):

    del self.__rnn

  def testARecurrentMNeuralNetwork(self): 

    fail = {}
    fail[735072] = datetime.strptime('2015-12-01', '%Y-%m-%d')
    fail[735076] = datetime.strptime('2016-06-02', '%Y-%m-%d')
    fail[735083] = datetime.strptime('2017-01-03', '%Y-%m-%d')
    fail[733548] = datetime.strptime('2017-04-10', '%Y-%m-%d')
    fail[733544] = datetime.strptime('2017-10-26', '%Y-%m-%d')
    fail[735087] = datetime.strptime('2017-12-12', '%Y-%m-%d')
    fail[735135] = datetime.strptime('2018-01-18', '%Y-%m-%d')
    fail[733642] = datetime.strptime('2018-01-29', '%Y-%m-%d')
    fail[735014] = datetime.strptime('2018-05-29', '%Y-%m-%d')
    fail[735155] = datetime.strptime('2018-06-26', '%Y-%m-%d')
    fail[733671] = datetime.strptime('2018-08-13', '%Y-%m-%d')

    data_file = DATA_DIR+'TAKEOFF_EGT_9_18_2018.csv'

    #self.__lclass.load_data(data_file, fail)

    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #print(y_train[10:])

    #input_dim = 784
    #nb_classes = 10
    #X_train = X_train.reshape(60000, input_dim)
    #X_test = X_test.reshape(10000, input_dim)
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')

    #X_train /= 255
    #X_test /= 255

    # convert class vectors to binary class matrices
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    #print(Y_train.shape)
    #print(Y_train)

    #self.__mnist.fit(X_train, Y_train, X_test, Y_test)
    #self.__mnist.save()

  def testBNASAChallenge(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'

    (X_train, y_train, X_test, y_test, events_train, events_test) = \
            self.__rnn.load_nasa_challenge_data(train_file, test_file)
    
    self.__rnn.fit(X_train, y_train, X_test, y_test, events_test)


def suite():
  suite = makeSuite(TestRNN, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
