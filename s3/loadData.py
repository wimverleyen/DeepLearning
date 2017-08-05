#!/usr/bin/python

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

from boto.s3.connection import S3Connection
#from odin_client import AWSCredentialsProvider, TimedRefresher
from boto.s3.key import Key

import json
import jsonschema
from pprint import pprint
from functools import partial
from multiprocessing import Pool

INPUT_DATA = "/home/ec2-user/work/data/"


def read_gz_file(filename):

  chunks = []
  for chunk in pd.read_csv(filename, chunksize=100000):
    chunk = chunk.apply(lambda x: x.replace('"', '')) 
    chunk.replace([np.inf, -np.inf], np.nan, inplace = True)
    chunk.fillna(0, inplace=True)
    chunks.append(chunk)
    #print set(chunk.dtypes.values)
    
  full = pd.concat(chunks)
  return full


def preprocess(filename, s3bucket):

  print filename
  key = s3bucket.get_key(filename)
  local_filename = INPUT_DATA+filename[(filename.rfind('/') + 1):]
  key.get_contents_to_filename(local_filename)
  partition = read_gz_file(local_filename)
  print partition.shape

  return partition


class S3:

  def __init__(self, S3BucketName, AWSAccessKeyID='...', \
                AWSSecretAccessKey='...'):

    self.__conn = S3Connection(AWSAccessKeyID, AWSSecretAccessKey)
    self.__s3bucket = self.__conn.get_bucket(S3BucketName)

  def __del__(self):

   del self.__s3bucket
   self.__conn.close() 

  def loadTarget(self, prefix, fileName):

    target_prefix = prefix + fileName 
    print target_prefix
    target_key = list(self.__s3bucket.list(prefix=target_prefix))[0]
    target_key = self.__s3bucket.get_key(target_key)
    local_filename = INPUT_DATA+fileName 
    target_key.get_contents_to_filename(local_filename)

    with open(local_filename) as target_file:    
      target = pd.read_csv(target_file)

    return target

  def getFilelist(self, prefix):

    objs = list(self.__s3bucket.list(prefix=prefix))
    filelist = [x.key for x in objs if 'part' in x.key]

    return filelist

  def loadPartition(self, fileName):

    partition = preprocess(fileName, self.__s3bucket)
    return partition

  def concat(self, prefix):

    objs = list(self.__s3bucket.list(prefix=prefix))
    filelist = [x.key for x in objs if x.key[-3:] == '.gz']
 
    pool = Pool(processes=3)
    result = pool.map(partial(preprocess, s3bucket=self.__s3bucket), [filename for filename in filelist])

    data = pd.concat(result)

    return data

  def features(self, prefix, dates):

    features = {}
    for date in dates:

      print "____ date= ", date
      features[date] = []

      json_prefix = prefix + "subsample_" + date + "/schema.json"
      json_key = list(self.__s3bucket.list(prefix=json_prefix))[0]
      json_key = self.__s3bucket.get_key(json_key)
      local_filename = '/home/ec2-user/work/data/schema_'+date+'.json'
      json_key.get_contents_to_filename(local_filename)

      with open(local_filename) as json_file:    
        data = json.load(json_file)
        for feature in data['fields']:
          if feature['name'] not in features[date]:
            features[date].append(feature['name'])

    all_features = []          
    for date in dates:
      print len(features[date])

  def metadata(self, prefix, dates):

    features = {}

    for date in dates:

      print "____ date= ", date
      mapping_prefix = prefix + "subsample_" + date + "/metadata/mapping.csv.gz"
      mapping_key = list(self.__s3bucket.list(prefix=mapping_prefix))[0]
      mapping_key = self.__s3bucket.get_key(mapping_key)
      local_filename = '/home/ec2-user/work/data/mapping_'+date+'.csv.gz'
      mapping_key.get_contents_to_filename(local_filename)
      mapping = pd.read_csv(local_filename)
      
      sparsity_prefix = prefix + "subsample_" + date + "/metadata/sparsity_report.csv.gz"
      sparsity_key = list(self.__s3bucket.list(prefix=sparsity_prefix))[0]
      sparsity_key = self.__s3bucket.get_key(sparsity_key)
      local_filename = '/home/ec2-user/work/data/sparsity_report_'+date+'.csv.gz'
      sparsity_key.get_contents_to_filename(local_filename)
      sparsity = pd.read_csv(local_filename)
      sparsity.sort(['sparsity'], ascending=False, inplace=True)

      features[date] = mapping.model_name.tolist()

    final = []
    i = 0
    for date in dates:

      if i == 0:
        final = features[date]
      else:
        final = list(set(final).intersection(set(features[date])))
      i += 1
      
    d = {}
    d['model_name'] = final
    df = pd.DataFrame(data=d)
    df.to_csv(INPUT_DATA+"Features_overlap.csv", index=False)

  def loadWeek(self, date, prefix, targetPrefix, targetFileName, features):

    features = pd.read_csv(features)
    features = features.model_name.tolist()
    features.remove('customer_id')

    target = self.loadTarget(targetPrefix, targetFileName)
    target.drop_duplicates(['customer_id'], keep='last', inplace=True)
    target.drop('hit_day', axis=1, inplace=True)

    filelist = self.getFilelist(prefix)

    partitions = []

    for filename in filelist:

      partition = self.loadPartition(filename)
      partition.drop_duplicates(['customer_id'], keep='last', inplace=True)

      partition = pd.merge(target, partition, how='inner', on='customer_id')
      partition.drop('customer_id', axis=1, inplace=True)
      print partition.shape

      partitions.append(partition)

    df = pd.concat(partitions)
    del partitions

    X = df[features]
    y = np.array(df['label']).ravel()
    del df

    return (X, y)

  def loadTrain(self, dates, prefix, targetPrefix, targetFileName, features):

    features = pd.read_csv(features)
    features = features.model_name.tolist()
    features.remove('customer_id')

    filelist = []
    targets = {}

    for date in dates:

      prefix_date = prefix+date+'/'
      target_file_name = targetFileName+date+'.csv'
      print targetPrefix
      print targetFileName

      target = self.loadTarget(targetPrefix, target_file_name)
      target.drop_duplicates(['customer_id'], keep='last', inplace=True)
      target.drop('hit_day', axis=1, inplace=True)
      targets[date] = target
      for file_name in self.getFilelist(prefix_date):
        filelist.append(file_name)

    partitions = []

    for filename in filelist:

      partition = self.loadPartition(filename)

      for date in dates:
        if date in filename:
          partition.drop_duplicates(['customer_id'], keep='last', inplace=True)
          partition = pd.merge(targets[date], partition, how='inner', on='customer_id')
          partition.drop('customer_id', axis=1, inplace=True)
          partitions.append(partition)

    df = pd.concat(partitions)
    del partitions

    X = df[features]
    y = np.array(df['label']).ravel()
    del df

    return (X, y)



class TestS3(TestCase):                                                                                                         
    
  def setUp(self):

    self.__s3 = S3('audible-ds-gen-data')
    
  def tearDown(self):

    del self.__s3

  def testAReadS3Bucket(self):

    prefix = 'data/wverleye/alchemy/twister/training/subsample_2017-02-20/'
    #data = self.__s3.concat(prefix)
    #print data.shape

  def testBReadS3Bucket(self):

    prefix = 'data/wverleye/alchemy/twister/training/'
    dates = ['2017-02-20', '2017-02-27', '2017-03-06']
    #self.__s3.features(prefix, dates)
    
    prefix = 'data/wverleye/alchemy/twister/training/'
    dates = ['2017-02-20', '2017-02-27', '2017-03-06', '2017-03-13', '2017-03-20']
    #self.__s3.metadata(prefix, dates)

  def testCReadS3Bucket(self):

    #prefix = 'data/wverleye/alchemy/twister/training/'
    prefix = 'data/wverleye/alchemy/twister/training/subsample_2017-02-20/'
    filelist = self.__s3.getFilelist(prefix)
    print filelist

    filename = 'part-00000.gz'
    partition = self.__s3.loadPartition(filelist[0])
    partition.drop_duplicates(['customer_id'], keep='last', inplace=True)
    print partition.shape

    prefix = 'wverleye/alchemy/amzsearch/us/'
    filename = 'Target_All_subsample_2017-02-20.csv'
    target = self.__s3.loadTarget(prefix, filename)
    target.drop_duplicates(['customer_id'], keep='last', inplace=True)
    print target.shape

    df = pd.merge(target, partition, how='inner', on='customer_id')
    print "merged: ", df.shape

  def testDLoadWeekS3(self):

    date = '2017-02-20'
    prefix = 'data/wverleye/alchemy/twister/training/subsample_2017-02-20/'
    target_prefix = 'wverleye/alchemy/amzsearch/us/'
    target_filename = 'Target_All_subsample_2017-02-20.csv'
    features = INPUT_DATA+'Features_overlap.csv'

    #(X, y) = self.__s3.loadWeek(date, prefix, target_prefix, target_filename, features)
    #print X.shape
    #print y.shape

  def testELoadTrainS3(self):

    print "Test load train"

    dates = ['2017-02-20', '2017-02-27', '2017-03-06']
    prefix = 'data/wverleye/alchemy/twister/training/subsample_'
    target_prefix = 'wverleye/alchemy/amzsearch/us/'
    target_filename = 'Target_All_subsample_'
    features = INPUT_DATA+'Features_overlap.csv'
    (X, y) = self.__s3.loadTrain(dates, prefix, target_prefix, target_filename, features)
    #print X.shape
    #print y.shape


def suite():
  suite = makeSuite(TestS3, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
