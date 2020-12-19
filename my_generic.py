import numpy as np
import math
import random

class generic():
    def __init__(self) -> None:
        super().__init__()
        self.num_population=100#随机生成的初始解的总数
        self.copy_num=70#保留的解的个数
        self.cross_num=20#交叉解的个数
        self.var_num=10#变异解的个数
        
    def initPopulation(self, cities : np.ndarray):
        self.num_cities = cities.shape[0]
        # generate distance matrix
        coord_x = cities[:,(0,)]
        coord_y = cities[:,(1,)]
        self.dist_mat = np.sqrt((coord_x - coord_x.T)**2 + (coord_y - coord_y.T)**2)
        # initialize first generation population 
        # @population ndarray: num_cities * [fitness, cities order]
        random_idx = np.random.permutation(np.random.arange(self.num_cities))
        self.population = np.zeros(self.num_population, self.num_cities+1)
        for i in range(self.num_population):
            random_idx = np.random.permutation(np.random.arange(self.num_cities))
            self.population[i, 0] = calcDist(random_idx, self.dist_mat)
            self.population[i, 1:] = random_idx
    
    def crossOver(self):
        pass
    
    def mutate(self):
        pass
    
    def select(self):
        pass
        
        
def calcDist(cities : np.ndarray, distMat : np.ndarray) -> int:
    assert(cities.shape[1]==distMat.shape[0]==distMat.shape[1])
    idx1 = cities
    idx2 = np.concatenate(idx1[1:], idx1[0])
    dist = distMat[idx1, idx2]
    return np.sum(dist)
    


def main():
    GA = generic()
    