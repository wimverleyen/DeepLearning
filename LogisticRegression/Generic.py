#!/usr/bin/python

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import l1_l2

from performance.metrics import Metrics

INPUT_DATA = "/home/ec2-user/work/data/"


class Generic:

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

  def loadData(self):

    train = pd.read_csv(INPUT_DATA+"train_v4_8000.csv") 
    test = pd.read_csv(INPUT_DATA+"test_v4_8000.csv")

    #features = train.columns.tolist()
    #features.remove('label')
    #features.remove('customer_id')
    #features.remove('date')
    #features.remove('index')

    mapping = pd.read_csv(INPUT_DATA+"mapping_audible_final_stat_manual.csv")
    features = mapping.model_name.tolist()

    X_train = np.asarray(train[features])
    y_train = np.asarray(train['label']).ravel()

    X_test = np.asarray(test[features])
    y_test = np.asarray(test['label']).ravel()

    # Scale data
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)
    del standard_scaler

    return (X_train, y_train), (X_test, y_test)

  def buildModel(self):

    self.__model = Sequential()
    if self.__classes == 2:
      self.__model.add(Dense(1, input_dim=self.__input_dim, activation='sigmoid'))
      #self.__model.add(Dense(1, input_dim=self.__input_dim, activation='sigmoid', W_regularizer=l1_l2(l1=0.1, l2=0.1)))
    else:
      self.__model.add(Dense(1, input_dim=self.__input_dim, activation='softmax'))

  def fit(self, X, y, X_test, y_test):

    self.buildModel()
    self.__model.summary()

    if self.__classes == 2:
      self.__model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
      #self.__model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
      self.__model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = self.__model.fit(X, y,
                    batch_size=self.__batch, epochs=self.__epochs,
                    verbose=1)#, validation_data=(X_test, y_test.ravel()))
                    #verbose=1, validation_data=(X_test, y_test.ravel()))
    score = self.__model.evaluate(X_test, y_test, verbose=0)

    metric = Metrics()
    if self.__classes == 2:
      y_hat_train = self.__model.predict_proba(X)
      y_hat_test = self.__model.predict_proba(X_test)
      print "AUROC= %.5lf" % (1-metric.AUROC(y, y_hat_train))
      print "AUROC= %.5lf" % (1-metric.AUROC(y_test, y_hat_test))
    else:
      y_hat_test = self.__model.predict_proba(X_test)
      for i in np.arange(0, self.__classes):
        print "i= %d; AUROC= %.5lf" % (i, 1-metric.AUROC(y_test[:,i], y_hat_test[:,i]))
    
    model = LogisticRegression(C = 10.0, penalty = 'l1', tol = 0.01).fit(X, y)
    y_hat = model.predict_proba(X_test)
    print "AUROC= %.5lf" % (1-metric.AUROC(y_test, y_hat[:, 1]))

    del metric

  def save(self):

    json_string = self.__model.to_json()
    open('mnist_Logistic_model.json', 'w').write(json_string)
    yaml_string = self.__model.to_yaml()
    open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

    # save the weights in h5 format
    self.__model.save_weights('mnist_Logistic_wts.h5')



class TestGeneric(TestCase):                                                                                                         
    
  def setUp(self):

    #self.__gen = Generic(1000, 2, 20, 8007)
    self.__gen = Generic(1000, 2, 20, 45)
    
  def tearDown(self):

    del self.__gen

  def testALogisticRegreesion(self): 

    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = self.__gen.loadData()

    print X_train.shape
    print y_train.shape
    #print y_train
    print type(y_train)

    print X_test.shape
    print type(y_test)

    self.__gen.fit(X_train, y_train, X_test, y_test)
    #iself.__gen.save()


def suite():
  suite = makeSuite(TestGeneric, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
