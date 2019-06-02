#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:09:17 2016
@author: Wim Verleyen
"""

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd
import tensorflow as tf

import pickle
from datetime import datetime

from scipy.stats import rankdata

from sklearn.preprocessing import StandardScaler, MinMaxScaler
  
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, merge, Activation, Add, BatchNormalization, Conv2D
from keras.utils import Sequence, np_utils
import numpy as np
import keras 
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
#DATA_DIR = "/Users/UCRP556/data/aviation/NASA/Challenge_Data/"

np.random.seed(813306)


class RULGenerator(Sequence):

  def __init__(self, X, y, ts=100):
    self.__X, self.__y = X, y
    self.__ts = ts

    batch_fixed = self.__y[self.__y <= ts]

    self.__batch_size = 3.0*(batch_fixed.shape[0])

  def __len__(self):
    #return int(np.ceil(len(self.__X) / float(self.__batch_size)))
    return 500 

  def __getitem__(self, idx):

    ix = np.where(self.__y>self.__ts)[0]
    iy = np.where(self.__y<=self.__ts)[0]

    Xx = self.__X[ix]
    RULx = self.__y[ix]
    Xy = self.__X[iy]
    RULy = self.__y[iy]

    ix_batch = np.random.permutation(np.arange(RULx.shape[0]))
    ix_batch = ix_batch[:(RULy.shape[0]*2)]

    Xx_batch = Xx[ix_batch]
    RULx_batch = RULx[ix_batch]
    X_batch = np.concatenate((Xx_batch, Xy), axis=0)
    RUL_batch = np.concatenate((RULx_batch, RULy), axis=0)
    return X_batch, RUL_batch



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


class ResNet:

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

  #def build_resnet(self, input_shape, n_feature_maps, nb_classes):
  def build_resnet(self, input_shape, n_feature_maps):

    x = Input(shape=(input_shape))
    conv_x = keras.layers.normalization.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, (8, 1), padding='same')(conv_x)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    conv_y = keras.layers.Conv2D(n_feature_maps, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1), padding='same')(x)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x)
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1), padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1), padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    y = Add()([shortcut_y, conv_z])
    y = Activation('relu')(y)
     
    full = keras.layers.pooling.GlobalAveragePooling2D()(y)   
    #out = Dense(nb_classes, activation='softmax')(full)
    out = Dense(1, activation='linear')(full)
    self.__model = Model(inputs=x, outputs=out)

  def build_model(self):

    self.__model = Sequential()
    self.__model.add(Dense(12, kernel_initializer='normal', bias_initializer='zeros', \
                            input_dim=self.__input_dim, activation='sigmoid'))

    self.__model.add(BatchNormalization())
    self.__model.add(Conv2D(n_feature_maps, 8, 1, border_mode='same'))
    self.__model.add(BatchNormalization())
    self.__model.add(Activation('relu'))
    
    self.__model.add(Conv2D(n_feature_maps, 5, 1, border_mode='same'))
    self.__model.add(BatchNormalization())
    self.__model.add(Activation('relu'))

    self.__model.add(Conv2D(n_feature_maps, 3, 1, border_mode='same'))
    self.__model.add(BatchNormalization())


    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
      self.__model.add(Conv2D(n_feature_maps, 1, 1, border_mode='same'))
      self.__model.add(BatchNormalization())
    else:
      self.__model.add(BatchNormalization())
    print('Merging skip connection')
    #y = merge([shortcut_y, conv_z], mode='sum')
    self.__model.add(Add())
    #y = Add()([shortcut_y, conv_z])
    #y = Activation('relu')(y)
    self.__model.add(Activation('relu'))

    full = keras.layers.pooling.GlobalAveragePooling2D()(y)   
    out = Dense(nb_classes, activation='softmax')(full)

  def load_nasa_challenge_data(self, train_file, test_file, train_gap=20, dev_file=''):

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

    if len(dev_file) > 0:

      df_dev = pd.read_csv(dev_file, sep=' ', header=None)
      df_dev.columns = columns
      df_dev.dropna(axis=1, inplace=True)
      df_dev['rank'] = df_dev.groupby('device_id')['cycles'].rank(ascending=False)
      df_dev['Y'] = np.zeros(df_dev.shape[0])
      df_dev.loc[df_dev['rank'] <= train_gap, ['Y']] = 1
      df_dev['RUL'] = np.zeros(df_dev.shape[0])

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

    if len(dev_file) > 0:
      dev = []
      for dev_id in df_dev['device_id'].unique():
        data = df_dev[df_dev['device_id'] == dev_id]
        data['RUL'] = data['cycles'].max() - data['cycles']
        dev.append(data)
      df_dev = pd.concat(dev, axis=0)

    parameters = ['cycles', 'setting1', 'setting2', 'setting3']
    #parameters = ['setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 21)]
    parameters += sensors

    df_train.sort_values(by=['cycles'], ascending=True, inplace=True)
    df_test.sort_values(by=['cycles'], ascending=True, inplace=True)
    if len(dev_file) > 0:
      df_dev.sort_values(by=['cycles'], ascending=True, inplace=True)

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

    if len(dev_file) > 0:
      d = {}
      d['device_id'] = []
      d['cycles'] = []
      for dev_id in df_dev['device_id'].unique():
        d['device_id'].append(dev_id)
        d['cycles'].append(df_dev[df_dev['device_id'] == dev_id]['cycles'].max())

      df_dev_events = pd.DataFrame(data=d)
      df_dev_events.sort_values(by=['cycles'], ascending=True, inplace=True)

    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(df_train[parameters].values)
    y_train = np.asarray(df_train['RUL']).ravel()
    #X_test = scaler.fit_transform(df_test[parameters].values)
    X_test = scaler.transform(df_test[parameters].values)
    y_test = np.asarray(df_test['RUL']).ravel()
    if len(dev_file) > 0:
      X_dev = scaler.transform(df_dev[parameters].values)
      y_dev = np.asarray(df_dev['RUL']).ravel()
      X_dev = X_dev.reshape(X_dev.shape + (1,1,))

    X_train = X_train.reshape(X_train.shape + (1,1,))
    X_test = X_test.reshape(X_test.shape + (1,1,))

    if len(dev_file) > 0:
      return X_train, y_train, X_test, y_test, df_train_events, df_test_events, X_dev, y_dev
    else:
      return X_train, y_train, X_test, y_test, df_train_events, df_test_events

  def fit(self, X, y, X_test, y_test, events, name='ResNet_NASA_Challenge', loss='mean_absolute_percentage_error'):

    self.build_resnet(X.shape[1:], 64)
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

  def save(self, name='ResNet_NASA_Challenge'):

    json_string = self.__model.to_json()
    handler = open(DATA_DIR+'model/'+name+'.json', 'w')
    handler.write(json_string)
    handler.close()
    yaml_string = self.__model.to_yaml()
    handler = open(DATA_DIR+'model/'+name+'.yaml', 'w')
    handler.write(yaml_string)
    handler.close()
    self.__model.save_weights(DATA_DIR+'model/'+name+'.h5')

  def test(self, X_test, y_test, name='ResNet_NASA_Challenge', loss='mean_absolute_percentage_error', test='test'):

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
    df.to_csv(DATA_DIR+'model/'+name+'_y_'+test+'.csv', index=False)

 
       
#def readucr(filename):
#    data = np.loadtxt(filename, delimiter = ',')
#    Y = data[:,0]
#    X = data[:,1:]
#    return X, Y
   
#nb_epochs = 1500
 
 
#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']



#flist  = ['/Users/UCRP556/data/open/Adiac/']
#for each in flist:
#    fname = each
#    x_train, y_train = readucr(fname+'Adiac_TRAIN')
#    x_test, y_test = readucr(fname+'Adiac_TEST')
#    nb_classes = len(np.unique(y_test))
#    batch_size = min(x_train.shape[0]/10, 16)
     
#    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
#    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
     
     
#    Y_train = np_utils.to_categorical(y_train, nb_classes)
#    Y_test = np_utils.to_categorical(y_test, nb_classes)
     
#    x_train_mean = x_train.mean()
#    x_train_std = x_train.std()
#    x_train = (x_train - x_train_mean)/(x_train_std)
      
#    x_test = (x_test - x_train_mean)/(x_train_std)
#    x_train = x_train.reshape(x_train.shape + (1,1,))
#    x_test = x_test.reshape(x_test.shape + (1,1,))
     
     
#    x , y = build_resnet(x_train.shape[1:], 64, nb_classes)
#    model = Model(input=x, output=y)
#    optimizer = keras.optimizers.Adam()
#    model.compile(loss='categorical_crossentropy',
#                  optimizer=optimizer,
#                  metrics=['accuracy'])
      
#    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
#                      patience=50, min_lr=0.0001) 
#    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
#              verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
#    log = pd.DataFrame(hist.history)
#    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])


class TestResNet(TestCase):

  def setUp(self):

    self.__resnet = ResNet(20, 1500, 25)
    
  def tearDown(self):

    del self.__resnet

  def testALogisticRegreesion(self): 

    train_file = DATA_DIR+'train.txt'
    test_file = DATA_DIR+'test.txt'
    dev_file = DATA_DIR+'final_test.txt'
    name = 'ResNet_NASA_Challenge_RUL_power_loss_a_2_3'

    resnet = ResNet(20, 1500, 25)
    (X, y, X_test, y_test, events_train, events_test, X_dev, y_dev) = \
            resnet.load_nasa_challenge_data(train_file, test_file, dev_file=dev_file)
    resnet.fit(X, y, X_test, y_test, events_test, name=name, loss=rul_power_loss(a_1=2, a_2=3))
    #resnet.fit_generator(X, y, X_test, y_test, events_test, name=name, loss=rul_power_loss(a_1=2, a_2=5))
    #resnet.save(name=name)
    #resnet.test(X_test, y_test, name=name, loss=rul_power_loss(a_1=2, a_2=3))
    del resnet



def suite():
  suite = makeSuite(TestResNet, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
