#!/usr/bin/python3

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/laptop/Documents/data/aviation/NASA/Challenge_Data/"
#DATA_DIR = "/Users/UCRP556/code/networks/data/NASA/Challenge_Data/"


def longitudinal_batch(data, labels, ts, batch_size, step=10, window=100):

  ix1 = np.where((data[:,data.shape[1]-1]>ts))
  ix2 = np.where((data[:,data.shape[1]-1]<ts+window))
  ix = np.intersect1d(ix1[0], ix2[0])
  #ix = np.where((data[:,data.shape[1]-1]>ts))
  data = data[ix,:]
  labels = labels[ix,:]
  data_size = data.shape[0]

  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
  print('batch/epoch', num_batches_per_epoch)

  def longitudinal_generator():
    
    i = 0
    while True:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
      shuffled_labels = labels[shuffle_indices]
      
      i += 1
      for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
        yield X, y

  return num_batches_per_epoch, longitudinal_generator()


def longitudinal_batch_train(data, labels, ts, batch_size):

  ix = np.where((data[:,data.shape[1]-1]<ts))
  data = data[ix]
  labels = labels[ix]
  data_size = data.shape[0]

  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
  print('batch/epoch', num_batches_per_epoch)

  def longitudinal_generator():
    
    while True:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
      shuffled_labels = labels[shuffle_indices]
      
      for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
        yield X, y

  return num_batches_per_epoch, longitudinal_generator()


def longitudinal_batch_test(data, labels, ts, batch_size):

  ix = np.where((data[:,data.shape[1]-1]>ts))
  data = data[ix]
  labels = labels[ix]
  data_size = data.shape[0]

  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
  print('batch/epoch', num_batches_per_epoch)

  def longitudinal_generator():
    
    while True:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
      shuffled_labels = labels[shuffle_indices]
      
      for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
        yield X, y

  return num_batches_per_epoch, longitudinal_generator()


def longitudinal_loss(events, epsilon=.001):
  """
    function closure:
    https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  """

  
  

  def loss(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), K.square(layer), axis=-1)
    return 0

  return loss





class LongitudinalClassification:

  def __init__(self, batch, classes, epochs, inputDim):

    self.__batch = batch
    self.__classes = classes
    self.__epochs = epochs
    self.__input_dim = inputDim

    self.__model = None

  def __del__(self):

    del self.__batch
    del self.__classes
    del self.__epochs
    del self.__input_dim
    del self.__model

  def build_model(self):

    self.__model = Sequential()
    #self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='softmax'))
    #self.__model.add(Dense(64, input_dim=self.__input_dim, activation='relu'))
    #self.__model.add(Dropout(.5))
    #self.__model.add(Dense(28, activation='relu'))
    #self.__model.add(Dropout(.5))
    #self.__model.add(Dense(2, activation='sigmoid'))
    self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='sigmoid'))

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

    #parameters = ['cycles', 'setting1', 'setting2', 'setting3']
    parameters = ['setting1', 'setting2', 'setting3']
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
    y_train = np.asarray(df_train['Y']).ravel()
    X_train = np.concatenate((X_train, df_train['cycles'].values.reshape((df_train['cycles'].values.shape[0], 1))), axis=1)

    X_test = scaler.fit_transform(df_test[parameters].values)
    y_test = np.asarray(df_test['Y']).ravel()
    X_test = np.concatenate((X_test, df_test['cycles'].values.reshape((df_test['cycles'].values.shape[0], 1))), axis=1)
    #np.concatenate((X_test, df_test['cycles'].values), axis=1)

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

    self.__model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    #self.__model.compile(optimizer='sgd', loss=custom_loss(events), metrics=['categorical_accuracy'])
    history = self.__model.fit(X, y, \
                    batch_size=self.__batch, epochs=self.__epochs, \
                    verbose=1, validation_data=(X_test, y_test))
    score = self.__model.evaluate(X_test, y_test, verbose=1)

    #metric = Metrics()
    #y_hat_test = self.__model.predict_proba(X_test)
    #for i in np.arange(0, self.__classes):
    #  print("i= %d; AUROC= %.5lf" % (i, 1-metric.AUROC(y_test[:,i], y_hat_test[:,i])))
    #del metric

  def fit_generator_split(self, X, y, X_test, y_test, ts=30):

    self.build_model()
    self.__model.summary()
   
    self.__model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    #self.__model.compile(optimizer='sgd', loss=longitudinal_loss(ts), metrics=['accuracy'])

    train_step = []
    train_batch = []
    test_step = []
    test_batch = []
    for i in np.arange(0, 30):
      ts_now = ts + i*(10)
      train_steps, train_batches = longitudinal_batch(X, y, ts_now, 100)
      test_steps, test_batches = longitudinal_batch(X_test, y_test, ts_now, 50)
      train_step.append(train_steps)
      train_batch.append(train_batches)
      test_step.append(test_steps)
      test_batch.append(test_batches)

    self.__model.fit_generator(train_batches, train_steps, epochs=20, validation_data=test_batches, \
                            validation_steps=test_steps) 

  def fit_generator(self, X, y, X_test, y_test, events):

    self.build_model()
    self.__model.summary()

    self.__model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    #self.__model.compile(optimizer='sgd', loss=longitudinal_loss(events), metrics=['accuracy'])

    train_step = []
    train_batch = []
    test_step = []
    test_batch = []
    #for row in events.tail(100).itertuples():
    for row in events.tail(100).iterrows():
      row = row[1]
      split = row[1]
      train_steps, train_batches = longitudinal_batch_train(X, y, split, 100)
      test_steps, test_batches = longitudinal_batch_test(X, y, split, 100)
      train_step.append(train_steps)
      train_batch.append(train_batches)
      test_step.append(test_steps)
      test_batch.append(test_batches)

    self.__model.fit_generator(train_batches, train_steps, epochs=20, validation_data=train_batch, \
                            validation_steps=train_step) 


  def save(self):
    json_string = self.__model.to_json()
    open('longitudinal_logistic_model.json', 'w').write(json_string)
    yaml_string = self.__model.to_yaml()
    open('longitudinal_logistic_model.yaml', 'w').write(yaml_string)

    # save the weights in h5 format
    self.__model.save_weights('longitudinal_logistic_wts.h5')



class TestLongitudinalClassification(TestCase):                                                                                                         
  def setUp(self):

    self.__lclass = LongitudinalClassification(20, 2, 20, 25)
    
  def tearDown(self):

    del self.__lclass

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
            self.__lclass.load_nasa_challenge_data(train_file, test_file)
    #input_dim = X_train.shape[1]
    nb_classes = 2
    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    #self.__lclass.fit(X_train, y_train, X_test, y_test, fail)

  def testCNASAChallenge(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'

    (X_train, y_train, X_test, y_test, events_train, events_test) = \
            self.__lclass.load_nasa_challenge_data(train_file, test_file)

    #input_dim = X_train.shape[1]
    nb_classes = 2
    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    #self.__lclass.fit(X_train, y_train, X_test, y_test, fail)
    #self.__lclass.fit_generator(X_train, y_train, X_test, y_test)
    self.__lclass.fit_generator(X_train, y_train, X_test, y_test, events_test)



def suite():
  suite = makeSuite(TestLongitudinalClassification, 'test')
  return suite


if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
