import sys
sys.path.append('../')

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential,clone_model, load_model
from keras.layers import Activation, Convolution2D, MaxPooling2D, Flatten, Dropout,Reshape
from keras.layers import Dense, LSTM
from keras.optimizers import SGD , Adam
from keras import losses
import random# -*- coding: utf-8 -*-


class policyGradient:
    
    def __init__(self, input_shape, output_shape, learning_rate, gamma ,batchSize,
                 iteration_save = 500, model_path = '', isReload = False):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.model_path = model_path
        self.isReload = isReload
        self.reward = 0
        self.iteration_save = iteration_save
        
    def loss_func(self, y_true, y_pred):
        
        cross_entropy = losses.categorical_crossentropy(y_true, y_pred[0])
        
        return -np.mean(cross_entropy* y_pred[1])
        
        
        
    def build_model(self, input_shape, output_shape):
        
        model = Sequential()
        model.add(Convolution2D(
                nb_filter = 64,
                kernel_size = 3,
                padding='same',
                input_shape=input_shape    
        ))
        model.add(Activation('selu'))
        model.add(Convolution2D(
                nb_filter = 64,
                kernel_size = 3,
                padding='same',
                input_shape=input_shape    
        ))
        model.add(Activation('selu'))
        
        model.add(MaxPooling2D(strides=1))
        
        model.add(Convolution2D(
                nb_filter = 128,
                kernel_size = 3,
                padding='same'        
        ))
        model.add(Activation('selu'))
            
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='selu'))
        model.add(Dense(output_shape))
        model.add(Activation('softmax'))
        
        adam = Adam(lr=self.learning_rate)
        model.compile(loss=self.loss_func,optimizer=adam)
        
        return model
    
    def action(self, state):
        action_prob = self.model.predict(state)
        action = np.random.choice(range(len(action_prob)), p=action_prob)  # select action w.r.t the actions prob
        
        return action
    
    def train(self, memory, iteration):
        
        state = [m[0] for m in memory]
        action = [m[1] for m in memory]
        reward = [m[2] for m in memory]
            
        discounted_reward = np.zeros_like(reward)
        running_reward = 0
        
        for t in reversed(range(0, len(self.ep_rs))):
            running_reward = running_reward * self.gamma + reward[t]
            discounted_reward[t] = running_reward
        
        #Normalize reward
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        
        self.model.fit(state, [action, reward])
        
        if iteration%self.iteration_save  == 0: 
             if self.model_path != '':
                 self.model.save(self.model_path)
                 print('Save model')
        
        
        
        
        
        