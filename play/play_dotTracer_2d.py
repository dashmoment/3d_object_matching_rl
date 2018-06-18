import sys
sys.path.append('../')
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import dotTracer_2d as env  
from agent import dqn 

env_o = env.dotTracer_2d(5) 

action_map = {
                0: [0,0],
                1: [1,0],
                2: [0,1],
                3: [-1,-1],
                4: [1,1],
                5: [-1,0],
                6: [0,-1],
                7: [1,-1],
                8: [-1,1]
              }
        
          
agent = dqn.dqn(
            input_shape = env_o.maze.shape, 
            output_shape = len(action_map),
            epsilon = 0.1, 
            min_epsilon = 0.001,
            GAMMA = 0.99, 
            learning_rate = 1e-4, 
            batchSize = 32,
            update_qmodel = 500,
            model_path = '', 
            isReload = False)


max_mameroy_size = 128
memory = []
loss_log = []
reward_log = []
cum_reward = 0

for iteration in tqdm(range(1000000)) :
    
    if len(memory) > max_mameroy_size:
        memory = memory[1:]
        
    state = copy.deepcopy(env_o.maze)    
    action = agent.action(state[np.newaxis,:])      
    state_t1, reward, isFinish = env_o.update(action_map[np.argmax(action)])        
    memory.append([state, np.argmax(action), reward, state_t1, isFinish])        
    
    '''
    print('***********', np.argmax(action) ,'*************')
    print(env_o.maze[:,:,0])      
    print('*******************************')
    '''
    
    if iteration > max_mameroy_size:     
        agent.train(memory, iteration)
        loss_log.append(agent.loss)
        cum_reward += reward 
        reward_log.append(cum_reward)   
        
        print('reward:{}, loss:{} '.format(reward, agent.loss))
        #_ = input("check")
    
    
plt.plot(reward_log)
plt.plot(loss_log)


