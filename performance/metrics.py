#!/usr/bin/python

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt
print "Switched to:",matplotlib.get_backend()

import seaborn as sns
sns.set(style="white", palette="muted")

import os
from scipy.stats import rankdata


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
    print df.head(n=10)

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


def suite():

  suite = makeSuite(TestMetrics, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
