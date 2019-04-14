#!/usr/bin/python3

from unittest import makeSuite, TestCase, main

import numpy as np
import pandas as pd

import os
import gym
import random

from datetime import datetime
from easy_tf_log import tflog

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import callbacks

from karpathy import prepro, discount_rewards

#from performance.metrics import Metrics

LOG_DATA = "/home/laptop/Documents/code/DL/DeepLearning/ReinforcementLearning/OnPolicy/log/"


class GradientPolicy:

  def __init__(self):

    self.__model = None

  def __del__(self):

    del self.__model

  def build_model(self):

    self.__model = Sequential()
    self.__model.add(Dense(units=200, input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
    self.__model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

  def fit(self):

    self.buildModel()
    self.__model.summary()

    self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def open_ai_pong(self):

    UP_ACTION = 2
    DOWN_ACTION = 3

    # hyperparameters
    gamma = .99

    # initialize variables
    resume = True
    running_reward = None
    epochs_before_saving = 10
    log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    # load pre-trained model if exist
    if (resume and os.path.isfile('my_model_weights.h5')):
      print("loading previous weights")
      model.load_weights('my_model_weights.h5')
                    
      # add a callback tensorboard object to visualize learning
      tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, \
                                         write_graph=True, write_images=True)

    # initializing environment
    env = gym.make('Pong-v0')

    observation = env.reset()

    # main loop
    for i in range(500):
      # render a frame
      env.render()

      # choose random action
      action = random.randint(UP_ACTION, DOWN_ACTION) 

      # run one step
      observation, reward, done, info = env.step(action)

      # if the episode is over, reset the environment
      if done:
        env.reset()

  def train(self):

    self.build_model()
    self.__model.summary()
    self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    UP_ACTION = 2
    DOWN_ACTION = 3

    # hyperparameters
    gamma = .99

    # initializing variables
    x_train, y_train, rewards = [],[],[]
    reward_sum = 0
    episode_nb = 0

    # initialize variables
    resume = True
    running_reward = None
    epochs_before_saving = 10
    log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    # load pre-trained model if exist
    if (resume and os.path.isfile(LOG_DIR+'my_model_weights.h5')):
      print("loading previous weights")
      self.__model.load_weights(LOG_DIR+'my_model_weights.h5')
                    
    # add a callback tensorboard object to visualize learning
    tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, \
                                       write_graph=True, write_images=True)

    # initializing environment
    env = gym.make('Pong-v0')
    observation = env.reset()
    prev_input = None

    # main loop
    while(True):

      # preprocess the observation, set input as difference between images
      cur_input = prepro(observation)
      x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
      prev_input = cur_input

      # forward the policy network and sample action according to the proba distribution
      proba = self.__model.predict(np.expand_dims(x, axis=1).T)
      action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
      y = 1 if action == 2 else 0 # 0 and 1 are our labels

      # log the input and label to train later
      x_train.append(x)
      y_train.append(y)

      # do one step in our environment
      observation, reward, done, info = env.step(action)
      rewards.append(reward)
      reward_sum += reward

      # end of an episode
      if done:
        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
                              
        # increment episode number
        episode_nb += 1
        # training
        self.__model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, callbacks=[tbCallBack], \
                         sample_weight=discount_rewards(rewards, gamma))
                                                                              
        # Saving the weights used by our model
        if episode_nb % epochs_before_saving == 0:    
          self.__model.save_weights('my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
                                                                                                                               # Log the reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        tflog('running_reward', running_reward, custom_dir=log_dir)

        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None


class TestGradientPolicy(TestCase):

  def setUp(self):

    self.__vgp = GradientPolicy()
    
  def tearDown(self):

    del self.__vgp

  def testAPongAIEnvironment(self): 

    self.__vgp.open_ai_pong()

  def testBTrainModel(self): 

    self.__vgp.train()

def suite():
  suite = makeSuite(TestGradientPolicy, 'test')
  return suite

if __name__ == "__main__":
  main(defaultTest='suite', argv=['-d'])
