import numpy as np
import math
import random

from numpy.core.numeric import cross

class generic():
    def __init__(self) -> None:
        super().__init__()
        self.num_population=100#随机生成的初始解的总数
        self.cross_num=20#交叉解的个数
        self.var_num=10#变异解的个数
        
    def initPopulation(self, citiesCoord : np.ndarray):
        self.num_cities = citiesCoord.shape[0]
        # generate distance matrix
        coord_x = citiesCoord[:,(0,)]
        coord_y = citiesCoord[:,(1,)]
        self.dist_mat = np.sqrt((coord_x - coord_x.T)**2 + (coord_y - coord_y.T)**2)
        # initialize first generation population 
        # @population ndarray: num_cities * [fitness, cities order]
        self.population = np.zeros(self.num_population, self.num_cities+1)
        for i in range(self.num_population):
            random_idx = np.random.permutation(np.arange(self.num_cities))
            self.population[i, 0] = calcFitness(random_idx, self.dist_mat)
            self.population[i, 1:] = random_idx
    
    def crossOver(self, crossRatio : float):
        assert(0<crossRatio<1)
        crossSpan = int(crossRatio * self.num_cities)
        children = np.zeros((self.cross_num, self.num_cities+1))
        for i in range(self.cross_num):
            parent_idx = np.random.choice(self.num_population, 2)
            parent1 = self.population[parent_idx[0]]
            parent2 = self.population[parent_idx[1]]
            
            child = 0
    
    def mutate(self):
        pass
    
    def select(self):
        pass
        
        
def calcFitness(citiesOrder : np.ndarray, distMat : np.ndarray) -> int:
    assert(citiesOrder.shape[1]==distMat.shape[0]==distMat.shape[1])
    idx1 = citiesOrder
    idx2 = np.concatenate(idx1[1:], idx1[0])
    dist = distMat[idx1, idx2]
    return 10000.0/np.sum(dist)
    


def main():
    GA = generic()
    