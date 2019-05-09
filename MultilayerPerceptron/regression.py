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
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
#DATA_DIR = "/Users/UCRP556/data/aviation/NASA/Challenge_Data/"


#def longitudinal_loss(ts, events, epsilon=.001):
def longitudinal_loss(events, epsilon=.001):
  """
    function closure:
    https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  """

  #print('ts= ', ts)
  
  

  def loss(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), K.square(layer), axis=-1)
    #print(confusion_matrix(y_true, y_pred))
    return K.mean(K.square(y_pred - y_true))

  return loss


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



class Regression:

  def __init__(self, batch, epochs, inputDim):

    self.__batch = batch
    self.__epochs = epochs
    self.__input_dim = inputDim

    self.__model = None
    self.__history = None

  def __del__(self):

    del self.__batch
    del self.__epochs
    del self.__input_dim
    del self.__model
    del self.__history

  def build_model(self):

    
    #self.__model.add(Dense(12, input_dim=self.__input_dim, activation='sigmoid'))
    #self.__model.add(Dense(6, activation='sigmoid'))
    #self.__model.add(Dense(4, activation='sigmoid'))

    self.__model = Sequential()
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='softmax'))
    self.__model.add(Dense(12, kernel_initializer='normal', bias_initializer='zeros', \
                            input_dim=self.__input_dim, activation='sigmoid'))
    #self.__model.add(Dropout(.2))
    self.__model.add(Dense(8, activation='sigmoid'))
    #self.__model.add(Dropout(.2))
    #self.__model.add(Dense(8, activation='sigmoid'))
    #self.__model.add(Dropout(.2))
    #self.__model.add(Dense(4, activation='sigmoid'))
    #self.__model.add(Dropout(.2))
    self.__model.add(Dense(4, activation='sigmoid'))
    #self.__model.add(Dropout(.2))

    #self.__model.add(Dense(1, activation='linear'))
    self.__model.add(Dense(1, activation='exponential'))
    #self.__model.add(Dense(1, activation='selu'))
    #self.__model.add(Dense(1, activation='elu'))
    #self.__model.add(Dense(1, input_dim=self.__input_dim, activation='linear'))
    #self.__model.add(Dense(1, input_dim=self.__input_dim, activation='exponential'))
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='exponential'))
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='elu'))
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='selu'))

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

    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(df_train[parameters].values)
    y_train = np.asarray(df_train['RUL']).ravel()
    X_test = scaler.fit_transform(df_test[parameters].values)
    y_test = np.asarray(df_test['RUL']).ravel()

    return X_train, y_train, X_test, y_test, df_train_events, df_test_events

  def load_data(self, data, events):

    #print(data)
    parameters = ['TAT', 'EGT', 'N2']

    df = pd.read_csv(data)
    #print(list(df.columns))
    #print(df.head(n=3))
    df = df[['ENGINE_SERIAL_NUMBER','CSR','RECORDED_DT','RWCTOMAR']+parameters].dropna()
    df['RECORDED_DT'] = pd.to_datetime(df['RECORDED_DT'])
    df = df.sort_values('RECORDED_DT')
    df.rename(columns = {'ESN':'ENGINE_SERIAL_NUMBER'}, inplace = True)

    #print(df.head(n=5))

  def fit(self, X, y, X_test, y_test, events, name='RNN_NASA_Challenge', loss='mean_absolute_percentage_error'):

    self.build_model()
    self.__model.summary()

    self.__model.compile(optimizer='rmsprop', loss=loss, metrics=['mae', 'acc'])
    #self.__model.compile(optimizer='sgd', loss=loss, metrics=['mae', 'acc'])
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

    #d = {}
    #d['y'] = y_test.ravel()
    #d['y_hat'] = y_hat_test.ravel()

    #df = pd.DataFrame(data=d)
    #df.to_csv(DATA_DIR+'model/'+name+'_y_test.csv', index=False)

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
    #self.__model.compile(optimizer='sgd', loss=loss, metrics=['mae', 'acc'])
    y_hat_test = self.__model.predict(X_test)

    d = {}
    d['y'] = y_test.ravel()
    d['y_hat'] = y_hat_test.ravel()

    df = pd.DataFrame(data=d)
    df.to_csv(DATA_DIR+'model/'+name+'_y_test.csv', index=False)

    


class TestRegression(TestCase):                                                                                                         
  def setUp(self):

    self.__reg = Regression(20, 2, 25)
    
  def tearDown(self):

    del self.__reg

  def testALogisticRegreesion(self): 

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
    name = 'MLP_NASA_Challenge_RUL_loss_a_7_5'

    #reg = Regression(20, 100, 25)
    #(X_train, y_train, X_test, y_test, events_train, events_test) = \
    #        reg.load_nasa_challenge_data(train_file, test_file)
    #reg.fit(X_train, y_train, X_test, y_test, events_test, name=name, loss=rul_loss(a_1=8, a_2=5))
    #reg.save(name=name)
    #reg.test(X_test, y_test, name=name, loss=rul_loss(a_1=8, a_2=5))
    #del reg

  def testCNASAChallenge(self):

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'
    name = 'MLP_NASA_Challenge_RUL_lin_loss_a_50_600'

    #reg = Regression(20, 100, 25)
    #(X_train, y_train, X_test, y_test, events_train, events_test) = \
    #        reg.load_nasa_challenge_data(train_file, test_file)
    #reg.fit(X_train, y_train, X_test, y_test, events_test, name=name, loss=rul_lin_loss(a_1=50, a_2=600))
    #reg.save(name=name)
    #reg.test(X_test, y_test, name=name, loss=rul_lin_loss(a_1=50, a_2=600))
    #del reg

  def testDNASAChallenge(self):

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'
    name = 'MLP_NASA_Challenge_RUL_power_loss_a_2_3'

    reg = Regression(20, 100, 25)
    (X_train, y_train, X_test, y_test, events_train, events_test) = \
            reg.load_nasa_challenge_data(train_file, test_file)
    reg.fit(X_train, y_train, X_test, y_test, events_test, name=name, loss=rul_lin_loss(a_1=2, a_2=3))
    reg.save(name=name)
    reg.test(X_test, y_test, name=name, loss=rul_lin_loss(a_1=50, a_2=600))
    del reg



def suite():
  suite = makeSuite(TestRegression, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
