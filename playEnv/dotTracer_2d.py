import sys
sys.path.append('../')
import numpy as np
import random

class dotTracer_2d:
    
    def __init__(self, mapSize):
        self.mapSize = mapSize
        self.maze = np.zeros((mapSize,mapSize), np.int32)
        self.__defaultSign = 0
        self.__playerSign = 10
        self.__targetSign = 100
        self.__finishReward = 100
        self.__moveReward = -10
        self.__obReward = -100
        
        self.restart()
        
    def restart(self):
        
        self.maze = np.zeros((self.mapSize, self.mapSize), np.int32)
        self.target = [random.randint(1, self.mapSize-2), random.randint(1, self.mapSize-2)]
        self.player = [random.randint(1, self.mapSize-2), random.randint(1, self.mapSize-2)]
        
        if np.equal(self.target, self.player).all():
            self.restart()
        
        self.maze[self.target[0], self.target[1]] = self.__targetSign
        self.maze[self.player[0], self.player[1]] = self.__playerSign
        
    def __isFinish(self):   
        if np.equal(self.target, self.player).all():
            return True
        else:
            return False
    
    def __isOutBound(self):
        
        if self.player[0] < 1 or self.player[0] >  self.mapSize-2 \
            or self.player[1] < 1 or self.player[1] >  self.mapSize-2:
                return True
        else:
            return  False
        
        
    def update(self, action):
        
        self.maze[self.player[0], self.player[1]] = self.__defaultSign
        self.player  = [self.player[0] + action[0], self.player[1] + action[1]]
        self.maze[self.player[0], self.player[1]] = self.__playerSign
        
        
        isFinish = self.__isFinish()
        isOB = self.__isOutBound()
        
        if isFinish:
            self.restart()
            print('Finish. Game Restarting')
            return  self.__finishReward, isFinish
        elif isOB:
            self.restart()
            print('Out of Bound. Game Restarting')
            return self.__obReward, isFinish
        else:
            return self.__moveReward, isFinish
            
        
#************Unitest*************

#env = dotTracer_2d(5)
#env.restart()
#
##*****restart test********** 
#for i in range(10000):
#    if np.equal(env.target, env.player).all():
#        print("target player overlap")
#    if env.target[0] < 1 or env.target[0] >  env.mapSize-2 \
#        or env.target[1] < 1 or env.target[1] >  env.mapSize-2 \
#        or env.player[0] < 1 or env.player[0] >  env.mapSize-2 \
#        or env.player[1] < 1 or env.player[1] >  env.mapSize-2:
#            print("init ponit outof bound")
#            print("Target:{}".format(env.target))
#            print("Player:{}".format(env.player))
#print(env.map)

#*******Play test***********
#env = dotTracer_2d(5)
#env.restart()
#print(env.map)
#while True:
#    x = input('action x:')
#    y = input('action y:')
#    
#    action = (int(y),int(x))
#    reward = env.update(action)
#    print(env.map)
#    print('********',reward,'*******')






