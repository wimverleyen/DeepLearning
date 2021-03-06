#!/usr/bin/python3

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd
import tensorflow as tf

import pickle
from datetime import datetime

from scipy.stats import rankdata

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
#DATA_DIR = "/Users/UCRP556/data/Aviation/Challenge_Data/"


def rul_power_loss(a_1=2, a_2=3):
  """
    function closure:
    https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  """

  def loss(y_true, y_pred):

    d = y_pred - y_true
    s = tf.where(d < 0, tf.math.pow(d, a_1), tf.math.pow(d, a_2))
    return K.sum(s)

  return loss


def rul_lin_loss(a_1=50, a_2=100):
  """
    function closure:
    https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  """


  def loss(y_true, y_pred):

    d = y_pred - y_true
    s = tf.where(d < 0, tf.math.multiply(-d, a_1), tf.math.multiply(d, a_2))
    return K.sum(s)

  return loss


def rul_loss(a_1=10, a_2=6):
  """
    function closure:
    https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  """


  def loss(y_true, y_pred):

    d = y_pred - y_true
    """
    dmin = tf.exp(-(d/float(a_1))) -1
    dmax = tf.exp(d/float(a_2)) - 1
    s = tf.zeros_like(d)
    flags = tf.math.greater(d, s)
    ix = tf.to_int32(tf.where(flags))
    s[ix] = dmin[ix]
    flags = tf.math.less(d, tf.zeros_like(d))
    ix = tf.to_int32(tf.where(flags))
    s[ix] = dmax[ix]
    """

    s = tf.where(d < 0,  tf.exp(-(d/float(a_1))) -1, tf.exp(d/float(a_2)) - 1)
    return K.sum(s)

  return loss


class RNN:

  def __init__(self, batch, epochs, inputDim, window):

    self.__batch = batch
    self.__epochs = epochs
    self.__input_dim = inputDim
    self.__window = window
    
    self.__model = None
    self.__history = None

  def __del__(self):

    del self.__batch
    del self.__epochs
    del self.__input_dim
    del self.__window
    del self.__model
    del self.__history

  def prepare_sequences(self, df):
    '''
    :param df:
    :return: 3D X_seq with shape (sequences, window, parameters) for X_seq, and a 1D y_seq with shape (sequences,)
    Steps:
      1. convert dataframe into X and Y arrays
      2. Transform X with scaler
      3. generate look back sequences with window length
    '''
    X = df[self.__x_params]
    X_scaled = pd.DataFrame(data=self.__scaler.transform(X), columns = self.__x_params)
    y = df[self.__y_params]

    # apply rolling window calculation if necessary
    if self.__rolling_window > 0:
      X_scaled = np.array(X_scaled.rolling(self.__rolling_window).mean().dropna())
      y = np.array(y.rolling(self.__rolling_window).mean().dropna())

    # generate sequences with window length
    X_seq = []
    y_seq = []
    n_rows = X_scaled.shape[0]
    for k in np.arange(n_rows - self.__window):
      X_seq.append(X_scaled[k: k + self.__window, :])

    for k in np.arange(n_rows - self.__window):
      y_seq.append(y[k + self.__window])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq

  def load_nasa_data_alternative(self, train_file, test_file):
    '''
    :param train_file: train.txt file
    :param test_file:  test.txt file
    :return: training and testing sequencess for keras model training
    '''
    #initialize columns and read dfs
    columns = ['device_id', 'cycles', 'setting1', 'setting2', 'setting3']
    sensors = ['sensor' + str(i + 1) for i in np.arange(0, 23)]
    columns += sensors

    df_train = pd.read_csv(train_file, sep=' ', header=None)
    df_train.columns = columns
    df_train.dropna(axis=1, inplace=True)

    df_test = pd.read_csv(test_file, sep=' ', header=None)
    df_test.columns = columns
    df_test.dropna(axis=1, inplace=True)

    df_train.sort_values(by=['device_id','cycles'], ascending=True, inplace=True)
    df_train.reset_index(inplace=True)

    df_test.sort_values(by=['device_id','cycles'], ascending=True, inplace=True)
    df_test.reset_index(inplace=True)

    #generate event df and RUL/Y columns for train and test
    df_train_events = df_train.groupby('device_id', as_index=False).agg({'cycles': 'max'})
    df_train_events.columns = ['device_id', 'max_cycles']
    df_train = df_train.merge(df_train_events,on='device_id')
    df_train['RUL'] = df_train['max_cycles'] - df_train['cycles']
    df_train['Y'] = np.where(df_train['RUL'] <= self.__train_gap, 1, 0)

    df_test_events = df_test.groupby('device_id', as_index=False).agg({'cycles': 'max'})
    df_test_events.columns = ['device_id', 'max_cycles']
    df_test = df_test.merge(df_test_events,on='device_id')
    df_test['RUL'] = df_test['max_cycles'] - df_test['cycles']
    df_test['Y'] = np.where(df_test['RUL'] <= self.__train_gap, 1, 0)

    #fit scaler on training set
    self.__scaler.fit(df_train[self.__x_params].values)

    #convert dataframe to sequences using prepare sequences function
    train_seq = [self.prepare_sequences(df=df_train[df_train['device_id']==id]) for id in df_train['device_id'].unique()]
    test_seq = [self.prepare_sequences(df=df_test[df_test['device_id']==id]) for id in df_test['device_id'].unique()]

    #convert sequences to np arrays
    X_train_seq = np.concatenate([tup[0] for tup in train_seq])
    y_train_seq = np.concatenate([tup[1] for tup in train_seq])

    X_test_seq = np.concatenate([tup[0] for tup in test_seq])
    y_test_seq = np.concatenate([tup[1] for tup in test_seq])

    #shuffle the training sequences if necessary
    if self.__shuffle_train:
      shuffle_index = np.random.permutation(X_train_seq.shape[0])
      X_train_seq[shuffle_index]
      y_train_seq[shuffle_index]

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, df_train_events, df_test_events
    
  def build_model_random(self):

    self.__model = Sequential()
   
    self.__model.add(LSTM(output_dim=50, input_shape=(self.__window, self.__input_dim), return_sequences=True))
    self.__model.add(Dropout(0.2))
    
    self.__model.add(LSTM(100, return_sequences=False))
    self.__model.add(Dropout(0.2))

    self.__model.add(Dense(output_dim=1))
    #self.__model.model.add(Activation("linear")) 
    self.__model.model.add(Activation("exponential")) 

  def build_model(self):

    self.__model = Sequential()
   
    self.__model.add(LSTM(output_dim=50, input_shape=(self.__window, self.__input_dim), return_sequences=True))
    self.__model.add(Dropout(0.2))
    
    self.__model.add(LSTM(100, return_sequences=False))
    self.__model.add(Dropout(0.2))

    self.__model.add(Dense(output_dim=1))
    self.__model.model.add(Activation("linear"))

    #self.__model = Sequential()
    #self.__model.add(Dense(16, input_dim=self.__input_dim, activation='sigmoid'))
    #model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    #self.__model.add(LSTM(4, input_shape=(50, 3, 1), return_sequences=True))
    #self.__model.add(LSTM(units=4, input_shape=(1, 1), stateful=True))
    #self.__model.add(LSTM(units=16, input_shape=(self.__window, self.__input_dim)))
    #self.__model.add(LSTM(units=16))
    #self.__model.add(LSTM(16, return_sequences=True))
    #self.__model.add(Dropout(0.2))
    #self.__model.add(LSTM(units=16))
    #model.add(TimeDistributed(Dense(vocabulary)))
    #self.__model.add(Activation('linear'))
    #self.__model.add(Dense(1, activation='linear'))

  def load_random_data(self):

    # samples, timestamps, and features
    X_train = np.random.random_sample((1125, 75, 2))
    y_train = np.random.random_sample((1125))

    X_test = np.random.random_sample((1125, 75, 2))
    y_test = np.random.random_sample((1125))

    return (X_train, y_train, X_test, y_test)

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

    train = []
    for dev_id in df_train['device_id'].unique():
      data = df_train[df_train['device_id'] == dev_id]
      data['RUL'] = data['cycles'].max() - data['cycles']
      train.append(data)
    df_train = pd.concat(train, axis=0)

    test = []
    for dev_id in df_test['device_id'].unique():
      data = df_test[df_test['device_id'] == dev_id]
      data['RUL'] = data['cycles'].max() - data['cycles']
      test.append(data)
    df_test = pd.concat(test, axis=0)

    """
    if len(dev_file) > 0:
      dev = []
      for dev_id in df_dev['device_id'].unique():
        data = df_dev[df_dev['device_id'] == dev_id]
        data['RUL'] = data['cycles'].max() - data['cycles']
        dev.append(data)
      df_dev = pd.concat(dev, axis=0)


    for dev_id in df_train['device_id'].unique():
      df_train.loc[df_train['device_id'] == dev_id, ['RUL']] = \
            df_train['cycles'].max() - df_train['cycles']

    for dev_id in df_test['device_id'].unique():
      df_test.loc[df_test['device_id'] == dev_id, ['RUL']] = \
            df_test['cycles'].max() - df_test['cycles']
    """

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

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(df_train[parameters].values)
    y_train = np.asarray(df_train['RUL']).ravel()
    X_test = scaler.fit_transform(df_test[parameters].values)
    y_test = np.asarray(df_test['RUL']).ravel()

    n = X_train.shape[0]
    
    X_train_seq = []
    y_train_seq = []
    for k in range(n - self.__window + 1):
      X_train_seq.append(X_train[k : k + self.__window])
    for k in np.arange(n - self.__window + 1):
      y_train_seq.append(y_train[k])

    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)

    n = X_test.shape[0]
    X_test_seq = []
    y_test_seq = []
    for k in range(n - self.__window + 1):
      X_test_seq.append(X_test[k : k + self.__window])
    for k in np.arange(n - self.__window + 1):
      y_test_seq.append(y_test[k])

    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, df_train_events, df_test_events

  def fit(self, X, y, X_test, y_test, name='RNN_NASA_Challenge', loss='mean_absolute_percentage_error'):

    self.build_model()
    self.__model.summary()

    #self.__model.compile(optimizer='sgd', loss='mean_absolute_percentage_error', metrics=['mae', 'acc'])
    self.__model.compile(optimizer='rmsprop', loss=loss, metrics=['mae', 'acc'])
    #self.__model.compile(optimizer='rmsprop', loss=rul_loss(), metrics=['mae', 'acc'])
    #self.__model.compile(optimizer='sgd', loss=longitudinal_loss(events), metrics=['categorical_accuracy'])
    self.__history = self.__model.fit(X, y, \
                    batch_size=self.__batch, epochs=self.__epochs, \
                    verbose=1, validation_data=(X_test, y_test))
    score = self.__model.evaluate(X_test, y_test, verbose=1)

    with open(DATA_DIR+'model/'+name+'_history.pkl', 'wb') as handler:
      pickle.dump(self.__history.history, handler)
    handler.close()

    #metric = Metrics()
    #y_hat_test = self.__model.predict(X_test)
    y_hat_train = self.__model.predict(X)
    y_hat_test = self.__model.predict(X_test)

  def fit_random(self, X, y, X_test, y_test):

    self.build_model_random()
    self.__model.summary()

    self.__model.compile(optimizer='rmsprop', loss='mean_absolute_percentage_error', metrics=['mae', 'acc'])
    #self.__model.compile(optimizer='sgd', loss=longitudinal_loss(events), metrics=['categorical_accuracy'])
    history = self.__model.fit(X, y, \
                    batch_size=self.__batch, epochs=self.__epochs, \
                    verbose=1, validation_data=(X_test, y_test))
    score = self.__model.evaluate(X_test, y_test, verbose=1)

    #metric = Metrics()
    #y_hat_test = self.__model.predict(X_test)
    y_hat_train = self.__model.predict(X)
    y_hat_test = self.__model.predict(X_test)

  def save(self, name='RNN_NASA_Challenge'):

    json_string = self.__model.to_json()
    handler = open(DATA_DIR+'model/'+name+'.json', 'w')
    handler.write(json_string)
    handler.close()
    yaml_string = self.__model.to_yaml()
    handler = open(DATA_DIR+'model/'+name+'.yaml', 'w')
    handler.write(yaml_string)
    handler.close()
    self.__model.save_weights(DATA_DIR+'model/'+name+'.h5')

  def test(self, X_test, y_test, name='RNN_NASA_Challenge', loss='mean_absolute_percentage_error'):

    with open(DATA_DIR+'model/'+name+'.json', 'r') as handler:
      json_string = handler.read()
      self.__model = model_from_json(json_string)
    handler.close()

    self.__model.load_weights(DATA_DIR+'model/'+name+'.h5')
    self.__model.compile(optimizer='rmsprop', loss=loss, metrics=['mae', 'acc'])
    print(X_test.shape)
    y_hat_test = self.__model.predict(X_test)

    d = {}
    d['y'] = y_test.ravel()
    d['y_hat'] = y_hat_test.ravel()

    df = pd.DataFrame(data=d)
    df.to_csv(DATA_DIR+'model/'+name+'_y_test.csv', index=False)




class TestRNN(TestCase):

  def setUp(self):

    self.__rnn = RNN(20, 10, 25, 75)
    
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

    #rnn = RNN(20, 6, 25, 75)
    #(X_train, y_train, X_test, y_test, events_train, events_test) = \
    #        rnn.load_nasa_challenge_data(train_file, test_file)
    #rnn.fit(X_train, y_train, X_test, y_test)
    #rnn.save()
    #rnn.test(X_test, y_test)
    #del rnn

  def testCNASAChallenge(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'

    name = 'RNN_NASA_Challenge_RUL'

    #rnn = RNN(20, 100, 25, 75)
    #(X_train, y_train, X_test, y_test, events_train, events_test) = \
    #        rnn.load_nasa_challenge_data(train_file, test_file)
    #rnn.fit(X_train, y_train, X_test, y_test, name=name)
    #rnn.save(name=name)
    #rnn.test(X_test, y_test, name=name)
    #del rnn

  def testDNASAChallenge(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'

    name = 'RNN_NASA_Challenge_rul_lin_loss_a_50_100'

    #rnn = RNN(20, 100, 25, 75)
    #(x_train, y_train, x_test, y_test, events_train, events_test) = \
    #        rnn.load_nasa_challenge_data(train_file, test_file)
    #rnn.fit(x_train, y_train, x_test, y_test, name=name, loss=rul_lin_loss(a_1=50, a_2=100))
    #rnn.save(name=name)
    #rnn.test(x_test, y_test, name=name, loss=rul_lin_loss(a_1=50, a_2=100))
    #del rnn

  def testENASAChallenge(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'

    name = 'RNN_NASA_Challenge_rul_power_loss_a_2_4'

    rnn = RNN(20, 100, 25, 75)
    (x_train, y_train, x_test, y_test, events_train, events_test) = \
            rnn.load_nasa_challenge_data(train_file, test_file)
    rnn.fit(x_train, y_train, x_test, y_test, name=name, loss=rul_power_loss(a_1=2, a_2=4))
    rnn.save(name=name)
    rnn.test(x_test, y_test, name=name, loss=rul_power_loss(a_1=2, a_2=4))
    del rnn  

  def testERandom(self):

    #rnn = RNN(20, 10, 2, 75)
    #(X_train, y_train, X_test, y_test) = rnn.load_random_data()
    #rnn.fit_random(X_train, y_train, X_test, y_test)
    #del rnn
    pass


def suite():
  suite = makeSuite(TestRNN, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
