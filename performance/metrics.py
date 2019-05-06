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


class Metrics:

  def __init__(self):
    pass

  def __del__(self):
    pass

  def _computeMetric(self, labels, scores, id, identifier, outputDir):
                                                                                                                                     
    idx = np.argsort(scores)

    labels = np.asarray(labels)
    labels = labels.ravel()
    precision = np.zeros(labels.shape[0])
    recall = np.zeros(labels.shape[0])
    fpr = np.zeros(labels.shape[0])
                                                                                                                                     
    sortedscores = scores[idx]
    sortedlabels = labels[idx]
    sorted_ids = id[idx]
                                                                                                                                     
    (Npos,) = np.where(labels == 1)[0].shape
    (Nneg,) = np.where(labels == 0)[0].shape
                                                                                                                                     
    i = 1
    for score in sortedscores:

      precision[i-1] = sortedlabels[0:i].sum()/float(i)
      recall[i-1] = sortedlabels[0:i].sum()/float(Npos)
      fpr[i-1] = (i - sortedlabels[0:i].sum())/float(Nneg)
                                                                                                                                     
      i += 1
                                                                                                                                     
    d = {'id': sorted_ids, 'label': sortedlabels, \
         'score': sortedscores, 'precision': precision, 'recall': recall, \
         'FPR': fpr}
    df = pd.DataFrame(data=d)
    
    df.to_csv(outputDir+'Precision_'+identifier+'.csv', index=False)
    d = {'id': sorted_ids, 'score': sortedscores}
    dfs = pd.DataFrame(data=d)
    dfs.to_csv(outputDir+'Scores_'+identifier+'.csv', index=False)

    return df

  def plotAUROCPrecision(self, labels, scores, id, identifier, outputDir):

    df = self._computeMetric(labels, scores, id, identifier, outputDir)
    print(df.head(n=10))

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)

    identity = np.arange(0, 1.1, 0.1)
    add_plot.plot(identity, identity, color='0.5', linewidth=.5)
    add_plot.plot(df['FPR'], df['recall'], color='forestgreen', linewidth=.75)
    
    label = "False positive rate"
    add_plot.set_xlabel(label, fontsize=12)
    label = "True positive rate"
    add_plot.set_ylabel(label, fontsize=12)
    auroc = self.AUROC(labels, scores)
    title = "ROC curve (area = %.4f)"%auroc
    add_plot.set_title(title, fontsize=14)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.grid(True)
    
    x = df['precision'].shape[0]
    xx = np.arange(0, x)
    xx = (xx/float(labels.shape[0]))*100
    
    add_plot = fig.add_subplot(1, 2, 2)
    
    add_plot.plot(xx, df['precision'], color='forestgreen', linewidth=.75)
    
    label = "% IDs"
    add_plot.set_xlabel(label, fontsize=12)
    add_plot.xaxis.set_ticks(np.arange(0, 110, 10))
    label = "Precision"
    add_plot.set_ylabel(label, fontsize=12)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.grid(True)
    
    filename = outputDir + "ROC_Curve_"+identifier+".png"
    fig.savefig(filename, format="png", dpi=300)

    del fig

  def AUROC(self, labels, scores):

    ranks = rankdata(scores)

    Npos = len([label for label in labels if label > 0])
    Nneg = len([label for label in labels if label <= 0])

    labels = np.asarray(labels)

    ranksum = 0.0
    index = 0
    for rank in ranks:
      if labels[index] == 1:                                                                                                    
        ranksum += rank                                                                                                              
      index += 1                                                                                                                     
    value = ranksum - ((Npos * (Npos + 1))/float(2))                                                                                 
    if Npos > 0:                                                                                                                     
      value = value/float(Npos * Nneg)                                                                                               
    else:                                                                                                                            
      value = 0.5                                                                                                                    
    return 1 - value

  def plot_loss_function(self, a_1=13, a_2=10, name='RNN_NASA_Challenge'):

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 1, 1)
  
    ref_1 = np.linspace(0, -50, 100)
    dmin = np.expm1(-(ref_1/float(a_1)))
    ref_2 = np.linspace(0, 50, 100)
    dmax = np.expm1(ref_2/float(a_2))

    ref_3 = np.linspace(0, -50, 100)
    lmin = np.expm1(-(ref_1/float(10)))
    ref_4 = np.linspace(0, 50, 100)
    lmax = np.expm1(ref_2/float(6))

    add_plot.plot(ref_1, dmin, color='indianred', linewidth=.75)
    add_plot.plot(ref_2, dmax, color='indianred', linewidth=.75, label='RUL score')
    #add_plot.scatter(d, s, color='dodgerblue', s=.75)
   
    add_plot.plot(ref_3, lmin, color='black', linewidth=.75, label='loss function')
    add_plot.plot(ref_4, lmax, color='black', linewidth=.75)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "Error"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "RUL score"
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)    

    figname = FIG_DIR +name+"_loss_function.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

  def plot_lin_loss_function(self, a_1_lin=50, a_2_lin=100, a_1=13, a_2=10, name='RNN_NASA_Challenge'):

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 1, 1)
  
    ref_1 = np.linspace(0, -50, 100)
    dmin = np.expm1(-(ref_1/float(a_1)))
    ref_2 = np.linspace(0, 50, 100)
    dmax = np.expm1(ref_2/float(a_2))

    ref_3 = np.linspace(0, -50, 100)
    lmin = np.expm1(-(ref_1/float(10)))
    ref_4 = np.linspace(0, 50, 100)
    lmax = np.expm1(ref_2/float(6))

    ref_5 = np.linspace(0, -50, 100)
    llmin = -np.multiply(ref_5, a_1_lin)
    ref_6 = np.linspace(0, 50, 100)
    llmax = np.multiply(ref_6, a_2_lin)


    add_plot.plot(ref_1, dmin, color='indianred', linewidth=.75)
    add_plot.plot(ref_2, dmax, color='indianred', linewidth=.75, label='RUL score')
    #add_plot.scatter(d, s, color='dodgerblue', s=.75)
   
    add_plot.plot(ref_3, lmin, color='black', linewidth=.75, label='loss function')
    add_plot.plot(ref_4, lmax, color='black', linewidth=.75)

    add_plot.plot(ref_5, llmin, color='dodgerblue', linewidth=.75, label='lin loss function')
    add_plot.plot(ref_6, llmax, color='dodgerblue', linewidth=.75)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "Error"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "RUL score"
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)    

    figname = FIG_DIR +name+"_lin_loss_function.png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

  def RUL_score(self, y, y_hat, a_1=13, a_2=10):

    d = y_hat - y
    dmin = np.expm1(-(d/float(a_1)))
    dmax = np.expm1(d/float(a_2))

    s = np.zeros(d.shape)

    ix = np.where(d < 0)[0]
    s[ix] = dmin[ix]

    ix = np.where(d >= 0)[0]
    s[ix] = dmax[ix]

    return s, s.sum()

  def plot_RUL(self, y, y_hat, name='RNN_NASA_Challenge', a_1=13, a_2=10, a_1_loss=8, a_2_loss=5):

    d = y_hat - y
    (s, score) = self.RUL_score(y, y_hat)

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)
  
    ref_1 = np.linspace(0, -50, 100)
    dmin = np.expm1(-(ref_1/float(a_1)))
    ref_2 = np.linspace(0, 50, 100)
    dmax = np.expm1(ref_2/float(a_2))

    ref_3 = np.linspace(0, -50, 100)
    lmin = np.expm1(-(ref_1/float(a_1_loss)))
    ref_4 = np.linspace(0, 50, 100)
    lmax = np.expm1(ref_2/float(a_2_loss))

    add_plot.plot(ref_1, dmin, color='indianred', linewidth=.75)
    add_plot.plot(ref_2, dmax, color='indianred', linewidth=.75, label='RUL score')
    add_plot.scatter(d, s, color='dodgerblue', s=.75)
   
    add_plot.plot(ref_3, lmin, color='black', linewidth=.75, label='loss function')
    add_plot.plot(ref_4, lmax, color='black', linewidth=.75)

    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "Error"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "RUL score"
    add_plot.set_ylabel(label, fontsize=10)
    title = "RUL score = %.2f" % score
    add_plot.set_title(title, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    identity = np.arange(0, 1.1, 0.1)
    #add_plot.plot(identity, identity, color='0.5', linewidth=.5)
    #add_plot.plot(y, d, color='forestgreen', linewidth=.75)
    add_plot.scatter(y, d, color='dodgerblue', s=.1, alpha=.7)
   
    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8)
    label = "RUL"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8)
    label = "Error"
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.grid(True)
    
    figname = FIG_DIR +name+".png"
    fig.savefig(figname, format="png", dpi=300)

    del fig

  def plot_learning(self, name='RNN_leearning', epochs=100):

    handler = open(DATA_DIR+'model/'+name+'_history.pkl', 'rb')
    history = pickle.load(handler)
    handler.close()

    df = pd.DataFrame(data=history)

    epochs = np.arange(0, epochs, 1)

    fig = plt.figure()

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.6, wspace=0.4)

    add_plot = fig.add_subplot(1, 2, 1)
  

    add_plot.semilogy(epochs, df['loss'].values, color='indianred', linewidth=.75, label='train')
    add_plot.semilogy(epochs, df['val_loss'].values, color='dodgerblue', linewidth=.75, label='test')
    #add_plot.plot(ref_2, dmax, color='indianred', linewidth=.75, label='RUL score')
    #add_plot.scatter(d, s, color='dodgerblue', s=.75)
   
    #add_plot.plot(ref_3, lmin, color='black', linewidth=.75, label='loss function')
    #add_plot.plot(ref_4, lmax, color='black', linewidth=.75)

    #add_plot.yscale('log')
    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "epochs"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8) 
    label = "loss"
    add_plot.set_ylabel(label, fontsize=10)
    #title = "RUL score = %.2f" % score
    #add_plot.set_title(title, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)

    add_plot = fig.add_subplot(1, 2, 2)

    identity = np.arange(0, 1.1, 0.1)
    #add_plot.plot(identity, identity, color='0.5', linewidth=.5)
    add_plot.plot(epochs, df['mean_absolute_error'].values, color='indianred', linewidth=.75, label='train')
    add_plot.plot(epochs, df['val_mean_absolute_error'].values, color='dodgerblue', linewidth=.75, label='test')
    #add_plot.scatter(y, d, color='dodgerblue', s=.1, alpha=.7)
  
    for tick in add_plot.xaxis.get_major_ticks():
      tick.label.set_fontsize(8)
    label = "epochs"
    add_plot.set_xlabel(label, fontsize=10)

    for tick in add_plot.yaxis.get_major_ticks():
      tick.label.set_fontsize(8)
    label = "MAE"
    add_plot.set_ylabel(label, fontsize=10)
    add_plot.set_aspect(1./add_plot.get_data_ratio())
    add_plot.legend(fontsize=8)
    add_plot.grid(True)
    
    figname = FIG_DIR + name+'_learning.png'
    fig.savefig(figname, format="png", dpi=300)

    del fig




class TestMetrics(TestCase):                                                                                                         
  def setUp(self):

    np.random.seed(600)
    self.__metric = Metrics()
    
  def tearDown(self):

    del self.__metric

  def testAROCPrecision(self): 

    labels = np.random.randint(0, 2, 3000)
    scores = np.random.uniform(0, 1, 3000)
    auroc = self.__metric.AUROC(labels, scores)
    self.assertTrue(0.5 <= auroc <= 0.6)

    id = np.arange(3000)
    output_dir = os.getcwd() + '/plotbin/'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    self.__metric.plotAUROCPrecision(labels, scores, id, "UnitTestA", output_dir)

  def testBRUL(self): 

    y = np.random.uniform(0, 1, 3000)
    y_hat = np.random.uniform(0, 1, 3000)
    #self.__metric.RUL_score(y, y_hat)

    name = 'RNN_NASA_Challenge_RUL'
    DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    filename = DATA_DIR+'model/'+name+'_y_test.csv'

    #df = pd.read_csv(filename)
    #print(df.head(n=5))

    #self.__metric.RUL_score(df['y'].values, df['y_hat'].values)
    #self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name)
    #self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name)

    """
    name = 'RNN_NASA_Challenge_RUL_loss'
    DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    filename = DATA_DIR+'model/'+name+'_y_test.csv'

    df = pd.read_csv(filename)
    #print(df.head(n=5))
    self.__metric.RUL_score(df['y'].values, df['y_hat'].values)
    self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name)
    self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name)

    name = 'RNN_NASA_Challenge_RUL_loss_as'
    DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    filename = DATA_DIR+'model/'+name+'_y_test.csv'
    df = pd.read_csv(filename)
    #print(df.head(n=5))
    self.__metric.RUL_score(df['y'].values, df['y_hat'].values)
    self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name)
    self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name)
    """

    name = 'RNN_NASA_Challenge_RUL_loss_a_10_6_exp'
    DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    filename = DATA_DIR+'model/'+name+'_y_test.csv'
    df = pd.read_csv(filename)

    #self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name+'_direct')
    #self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name) 

    #name = 'MLP_NASA_Challenge_RUL_loss'
    #DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    #filename = DATA_DIR+'model/'+name+'_y_test.csv'

    #df = pd.read_csv(filename)
    #print(df.head(n=5))

    #self.__metric.RUL_score(df['y'].values, df['y_hat'].values)
    #self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name)
    #self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name) 

    """
    name = 'MLP_NASA_Challenge_RUL_loss_as'
    DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    filename = DATA_DIR+'model/'+name+'_y_test.csv'

    df = pd.read_csv(filename)
    print(df.head(n=5))

    self.__metric.RUL_score(df['y'].values, df['y_hat'].values)
    self.__metric.plot_RUL(df['y'].values, df['y_hat'].values, name=name)
    self.__metric.plot_learning('RNN_NASA_Challenge_RUL_loss_history.pkl', name=name) 
    """

    #DATA_DIR = "/home/wimverleyen/data/aviation/NASA/Challenge_Data/"
    #filename = DATA_DIR+'model/'

    #df = pd.read_csv(filename)
    #print(df.head(n=5))

    #self.__metric.RUL_score(df['y'].values, df['y_hat'].values)

  def testCLossRUL(self): 

    self.__metric.plot_loss_function()
    self.__metric.plot_lin_loss_function()


def suite():

  suite = makeSuite(TestMetrics, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
