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
import random

class dqn:
    
    def __init__(self, input_shape, output_shape, 
                 epsilon, min_epsilon, GAMMA, learning_rate, batchSize, 
                 update_qmodel, model_path = '', isReload = False):
        
        self.output_shape = output_shape
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decayrate = (epsilon - self.min_epsilon)/200
        
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.loss = 0
        self.update_qmodel = update_qmodel
        self.model_path = model_path
        
        if not isReload:
            self.model = self.build_model(input_shape, output_shape)         
        elif isReload and model_path != '':
            self.model = load_model(model_path)
            
        self.model_q = clone_model(self.model)
            
    
    def build_model(self, input_shape, output_shape):
        
        model = Sequential()
        model.add(Convolution2D(
                nb_filter = 128,
                kernel_size = 5,
                padding='same',
                input_shape=input_shape    
        ))
        model.add(Activation('selu'))
        #model.add(MaxPooling1D(stride=1))
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
        model.add(Activation('linear'))
        
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        
        return model
    
    def action(self, state, istrain = True):
        
        #print(self.epsilon)
           
        action = np.zeros(self.output_shape)
        if  random.random() <= self.epsilon and istrain: 
            action[random.randrange(self.output_shape)] = 1
        else:       
            action[np.argmax(self.model.predict(state))] = 1
                  
        return action
    
    def train(self, memory, iteration): #memory: [ s, a, r , s']
        
         if self.epsilon >  self.min_epsilon:
             if iteration%500 == 0:
                 self.epsilon = self.epsilon - self.epsilon_decayrate
             
         minibatch = random.sample(memory, self.batchSize)
         state = [m[0] for m in minibatch]
         action = [m[1] for m in minibatch]
         reward = [m[2] for m in minibatch]
         state_t1 = [m[3] for m in minibatch]
         isfinish =  [m[4] for m in minibatch]
         
         state = np.stack(state, axis=0)
         state_t1 = np.stack(state_t1, axis=0)
         targets = self.model.predict(state)
         
         for i in range(len(minibatch)):
             
             if isfinish == 1:
                 targets[i, action[i]] = reward[i]
             else:
                 predict_t1 = self.model.predict(state_t1[i][np.newaxis, :] ) 
                 targets[i, action[i]] = reward[i] + self.GAMMA * np.max(predict_t1, axis=-1)
                 
         self.loss = self.model.train_on_batch(state, targets)
         
         if iteration%self.update_qmodel == 0: 
             self.model_q = clone_model(self.model)
             if self.model_path != '':
                 self.model.save(self.model_path)
                 print('Update Q network and save model')

