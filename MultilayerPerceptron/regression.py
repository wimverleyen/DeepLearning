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
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/laptop/Documents/data/aviation/NASA/Challenge_Data/"
#DATA_DIR = "/Users/UCRP556/code/networks/data/NASA/Challenge_Data/"

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



class Regression:

  def __init__(self, batch, epochs, inputDim):

    self.__batch = batch
    self.__epochs = epochs
    self.__input_dim = inputDim

    self.__model = None

  def __del__(self):

    del self.__batch
    del self.__epochs
    del self.__input_dim
    del self.__model

  def build_model(self):

    self.__model = Sequential()
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='softmax'))
    self.__model.add(Dense(12, input_dim=self.__input_dim, activation='sigmoid'))
    #self.__model.add(Dense(64, kernel_initializer='normal', bias_initializer='zeros', input_dim=self.__input_dim, activation='sigmoid'))
    self.__model.add(Dropout(.2))
    self.__model.add(Dense(6, activation='sigmoid'))
    #self.__model.add(Dense(32, activation='sigmoid'))
    #self.__model.add(Dropout(.5))
    #self.__model.add(Dense(16, activation='sigmoid'))
    #self.__model.add(Dropout(.5))
    self.__model.add(Dense(4, activation='sigmoid'))

    self.__model.add(Dense(1, activation='linear'))
    #self.__model.add(Dense(1, activation='exponential'))
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

    scaler = StandardScaler()
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

  def fit(self, X, y, X_test, y_test, events):

    self.build_model()
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
    #d = {}
    #d['y_hat_test'] = y_hat_test
    #d['y_test'] = y_test
    #df = pd.DataFrame(data=d)
    #df['rank'] = rankdata(y_hat_test)
    #df.sort_values(by=['y_hat_test'], ascending=False, inplace=True)
    #print(df.head(n=5))
    #print(df.tail(n=5))
    #for i in np.arange(0, self.__classes):
    #  print("i= %d; AUROC= %.5lf" % (i, 1-metric.AUROC(y_test[:,i], y_hat_test[:,i])))
    #del metric

  def save(self):

    json_string = self.__model.to_json()
    open('longitudinal_logistic_model.json', 'w').write(json_string)
    yaml_string = self.__model.to_yaml()
    open('longitudinal_logistic_model.yaml', 'w').write(yaml_string)

    # save the weights in h5 format
    self.__model.save_weights('longitudinal_logistic_wts.h5')



class TestRegression(TestCase):                                                                                                         
  def setUp(self):

    self.__reg = Regression(20, 500, 25)
    
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

    (X_train, y_train, X_test, y_test, events_train, events_test) = \
            self.__reg.load_nasa_challenge_data(train_file, test_file)
    #input_dim = X_train.shape[1]
    nb_classes = 2
    
    # convert class vectors to binary class matrices
    #y_train = np_utils.to_categorical(y_train, nb_classes)
    #y_test = np_utils.to_categorical(y_test, nb_classes)

    self.__reg.fit(X_train, y_train, X_test, y_test, events_test)


def suite():
  suite = makeSuite(TestRegression, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
