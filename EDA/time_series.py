#!/usr/bin/python

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

import pickle
import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt
#print "Switched to:",matplotlib.get_backend()

import seaborn as sns
sns.set(style="white", palette="muted")

import os
from scipy.stats import rankdata

DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
FIG_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/figures/"

class Fleet:
  def __init__(self):
    pass

  def __del__(self):
    pass


class TestFleet(TestCase):                                                                                                         
  def setUp(self):

    np.random.seed(600)
    self.__fleet = Fleet()
    
  def tearDown(self):

    del self.__fleet

  def testADevice(self):

    device_id = 2

    columns = ['device_id', 'cycles', 'setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 23)]
    columns += sensors

    test_file = DATA_DIR+'test.txt'
    df = pd.read_csv(test_file, sep=' ', header=None)
    df.columns = columns
    print(df.head(n=5))

    name = 'MLP_NASA_Challenge_RUL_sample_power_loss_a_2_4'

    filename = DATA_DIR+'model/'+name+'_y_test.csv'
    df_test = pd.read_csv(filename)

    df['y_hat'] = df_test['y_hat'].values

    data = df[df['device_id'] == device_id][['device_id', 'cycles', 'y_hat']]
    
    data['y'] = np.zeros(data.shape[0])
    data['y'] = data['cycles'].max() - data['cycles']
    print(data.head(n=5))
    print(data.shape)

    #self.__metric.plot_lifetime(data, name=name)

    data = df[df['device_id'] == device_id][['device_id', 'cycles', 'y_hat']+sensors]
    #data = df[df['device_id'] == device_id][['device_id', 'cycles', 'y_hat']+sensors]
    #df_sensors = df[sensors]
    for sensor in sensors:
      #self.__metric.plot_degradation(data, name=name+'_sensor')
      self.__metric.plot_degradation_fleet(df[['device_id', 'cycles', sensor]], sensor, name=name)
    
    #self.__metric.plot_pairplot_fleet(data, sensors, name=name)



def suite():

  suite = makeSuite(TestFleet, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
