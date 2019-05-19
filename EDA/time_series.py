#!/usr/bin/python

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

import os
import pickle

from scipy.stats import rankdata, mannwhitneyu

import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", palette="muted")


DATA_DIR = "/Users/UCRP556/data/aviation/NASA/Challenge_Data/"
FIG_DIR = "/Users/UCRP556/data/aviation/NASA/Challenge_Data/figures/"
REPORT_DIR = "/Users/UCRP556/data/aviation/NASA/Challenge_Data/reports/"

class Fleet:
  def __init__(self):
    pass

  def __del__(self):
    pass

  def _MC_degradation(self, df, sensor, length, N=1000):

    sample = df[sensor].values

    total_length = df.shape[0]
    reference = df[sensor].tail(n=length).values
    print(reference.shape)

    max_last = total_length - 2*length

    for i in np.arange(0, N):
      start = np.random.randint(0, max_last)
      sequence = sample[start:start+length]

      print(sequence.shape)
      u, p = mannwhitneyu(reference, sequence)
      print(p)


  def plot_pairplot_fleet(self, data, sensors, name='device_lifetime'):

    sns_pair = sns.pairplot(data[sensors])
    fig = sns_pair.get_figure()

    figname = FIG_DIR+name+"_sensor_pair.png"
    fig.savefig(figname, format="png", dpi=300)

  def plot_degradation_fleet(self, df, sensor, name='device_lifetime'):

    device_ids = df['device_id'].unique()

    events = {}
    events['device_id'] = device_ids
    df_events = pd.DataFrame(df.groupby('device_id')['cycles'].max())
    df_events.reset_index(inplace=True)

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)

    last_values = []
    cycles = []
    ids = []

    for device_id in device_ids:

      data = df[df['device_id'] == device_id]
      last_values.append(float(data[sensor].tail(n=1)))
      cycles.append(int(data['cycles'].tail(n=1)))
      ids.append(int(device_id))

      add_plot.scatter(data['cycles'].values, \
              data[sensor].values, color='dodgerblue', s=.05, alpha=.3)

    add_plot.scatter(cycles, last_values, color='indianred', s=.5)
    i = 0
    for x, y in zip(cycles, last_values):
      label = "{}".format(ids[i])
      add_plot.annotate(label, (x, y), textcoords="offset points", \
                xytext=(0,10), ha='center', fontsize=4)
      i += 1

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    for device_id in device_ids:

      data = df[df['device_id'] == device_id]
      data[sensor+'_MA'] = data[sensor].rolling(75, center=True).mean()
      #data[sensor+'_delta'] = np.log(data[sensor]/data[sensor].shift())
      #data[sensor+'_volatility'] = data[sensor+'_delta'].rolling(31).std().shift()
      add_plot.plot(data[sensor+'_MA'].values, color='dodgerblue', linewidth=.1)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    figname = FIG_DIR+'fleet/'+name+'_'+sensor+"_fleet_degradation.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

  def plot_volatility_fleet(self, df, sensor, name='device_lifetime'):

    device_ids = df['device_id'].unique()

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)
    for device_id in device_ids:

      data = df[df['device_id'] == device_id]
      data[sensor+'_delta'] = np.log(data[sensor]/data[sensor].shift())

      #add_plot.scatter(data['cycles'].values, \
      #        data[sensor+'_delta'].values, color='dodgerblue', s=.05, alpha=.3)
      add_plot.plot(data['cycles'].values, \
              data[sensor+'_delta'].values, color='dodgerblue', linewidth=.1)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    for device_id in device_ids:

      data = df[df['device_id'] == device_id]
      data[sensor+'_delta'] = np.log(data[sensor]/data[sensor].shift())
      data[sensor+'_volatility'] = data[sensor+'_delta'].rolling(31).std().shift()
      add_plot.plot(data[sensor+'_volatility'].values, color='dodgerblue', linewidth=.1)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    figname = FIG_DIR+'fleet/'+name+'_'+sensor+"_fleet_volatility.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

    df.sort_values(by=['cycles'], inplace=True, ascending=False)
    print(df.head(n=5))

  def report_events(self, df, name='device_lifetime'):
    device_ids = df['device_id'].unique()

    events = {}
    events['device_id'] = device_ids
    df_events = pd.DataFrame(df.groupby('device_id')['cycles'].max())
    df_events.reset_index(inplace=True)
    df_events.sort_values(by=['cycles'], inplace=True, ascending=False)

    df_events.to_csv(REPORT_DIR+name+'_events_report.csv', index=False)

  def plot_degradation_device(self, df, sensor, name='device_lifetime', length=75):

    device_id = df['device_id'].unique()

    self._MC_degradation(df, sensor, length)

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)

    add_plot.plot(df['cycles'].values, \
              df[sensor].values, color='dodgerblue', linewidth=.5, alpha=.3)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    figname = FIG_DIR+'fleet/'+name+'_'+sensor+"_device_"+str(int(device_id))+"_degradation.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

  def plot_volatility_device(self, df, sensor, name='device_lifetime'):

    device_id = df['device_id'].unique()

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)

    df[sensor+'_delta'] = np.log(df[sensor]/df[sensor].shift())
    df[sensor+'_volatility'] = df[sensor+'_delta'].rolling(31).std().shift()
    add_plot.plot(df['cycles'].values, \
              df[sensor+'_volatility'].values, color='dodgerblue', linewidth=.5, alpha=.3)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "cycles"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = sensor
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    add_plot.hist(df[sensor+'_volatility'].values, color='dodgerblue', alpha=.5)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "volatility"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = '#'
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    figname = FIG_DIR+'fleet/'+name+'_'+sensor+"_device_"+str(int(device_id))+"_volatility.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig


class TestFleet(TestCase):                                                                                                         
  def setUp(self):

    np.random.seed(600)
    self.__fleet = Fleet()
    
  def tearDown(self):

    del self.__fleet

  def testAFleet(self):

    device_id = 2

    columns = ['device_id', 'cycles', 'setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 23)]
    columns += sensors

    test_file = DATA_DIR+'test.txt'
    df = pd.read_csv(test_file, sep=' ', header=None)
    df.columns = columns
    print(df.head(n=5))

    data = df[df['device_id'] == device_id][['device_id', 'cycles']]
    
    print(data.head(n=5))
    print(data.shape)

    #self.__metric.plot_lifetime(data, name=name)

    data = df[df['device_id'] == device_id][['device_id', 'cycles']+sensors]
    name = 'MLP_NASA_Challenge_RUL_sample_power_loss_a_2_4'
    #data = df[df['device_id'] == device_id][['device_id', 'cycles', 'y_hat']+sensors]
    #df_sensors = df[sensors]
    #for sensor in sensors:
      #self.__metric.plot_degradation(data, name=name+'_sensor')
    #  self.__metric.plot_degradation_fleet(df[['device_id', 'cycles', sensor]], sensor, name=name)
    
    #self.__metric.plot_pairplot_fleet(data, sensors, name=name)
    #self.__fleet.plot_degradation_fleet(df[['device_id', 'cycles', 'sensor9']], \
    #        'sensor9', name=name)

    #self.__fleet.plot_volatility_fleet(df[['device_id', 'cycles', 'sensor9']], \
    #        'sensor9', name=name)

    #self.__fleet.report_events(df[['device_id', 'cycles']], name=name)

  def testBDevice(self):

    device_id = 31

    columns = ['device_id', 'cycles', 'setting1', 'setting2', 'setting3']
    sensors = ['sensor'+str(i+1) for i in np.arange(0, 23)]
    columns += sensors

    test_file = DATA_DIR+'test.txt'
    df = pd.read_csv(test_file, sep=' ', header=None)
    df.columns = columns


    data = df[df['device_id'] == device_id][['device_id', 'cycles']+sensors]
    name = 'MLP_NASA_Challenge_RUL_sample_power_loss_a_2_4'
    
    self.__fleet.plot_degradation_device(data[['device_id', 'cycles', 'sensor9']], \
            'sensor9', name=name)

    #self.__fleet.plot_volatility_device(data[['device_id', 'cycles', 'sensor9']], \
    #        'sensor9', name=name)



def suite():

  suite = makeSuite(TestFleet, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
