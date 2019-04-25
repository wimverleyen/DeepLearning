#!/usr/bin/python3

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K

#from performance.metrics import Metrics

DATA_DIR = "/home/laptop/Documents/data/aviation/OTD/"

def custom_loss(events):
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

  def buildModel(self):

    self.__model = Sequential()
    self.__model.add(Dense(self.__classes, input_dim=self.__input_dim, activation='softmax'))

  def load_data(self, data, events):

    print(data)
    parameters = ['TAT', 'EGT', 'N2']

    df = pd.read_csv(data)
    print(list(df.columns))
    print(df.head(n=3))
    df = df[['ENGINE_SERIAL_NUMBER','CSR','RECORDED_DT','RWCTOMAR']+parameters].dropna()
    df['RECORDED_DT'] = pd.to_datetime(df['RECORDED_DT'])
    df = df.sort_values('RECORDED_DT')
    df.rename(columns = {'ESN':'ENGINE_SERIAL_NUMBER'}, inplace = True)

    print(df.head(n=5))

    pass

  def fit(self, X, y, X_test, y_test, events):

    self.buildModel()
    self.__model.summary()

    #self.__model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    self.__model.compile(optimizer='sgd', loss=custom_loss(events), metrics=['categorical_accuracy'])
    history = self.__model.fit(X, y,
                    batch_size=self.__batch, nb_epoch=self.__epochs,
                    verbose=1, validation_data=(X_test, y_test))
    score = self.__model.evaluate(X_test, y_test, verbose=0)

    #metric = Metrics()
    #y_hat_test = self.__model.predict_proba(X_test)
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



class TestLongitudinalClassification(TestCase):                                                                                                         
  def setUp(self):

    self.__lclass = LongitudinalClassification(128, 10, 20, 784)
    
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

    self.__lclass.load_data(data_file, fail)

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


def suite():
  suite = makeSuite(TestLongitudinalClassification, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])




