import numpy as np


class generic():
    def __init__(self) -> None:
        super().__init__()
        self.num_population=100#随机生成的初始解的总数
        self.cross_num=20#交叉解的个数
        self.mutate_num=10#变异解的个数
        
    def initPopulation(self, citiesCoord : np.ndarray):
        self.num_cities = citiesCoord.shape[0]
        # generate distance matrix
        coord_x = citiesCoord[:,(0,)]
        coord_y = citiesCoord[:,(1,)]
        self.dist_mat = np.sqrt((coord_x - coord_x.T)**2 + (coord_y - coord_y.T)**2)
        # initialize first generation population 
        # @population ndarray: num_cities * [fitness, cities order]
        self.population = np.zeros((self.num_population, self.num_cities+1))
        for i in range(self.num_population):
            random_idx = np.random.permutation(np.arange(self.num_cities))
            self.population[i, 0] = self.calcFitness(random_idx)
            self.population[i, 1:] = random_idx
    
    def crossOver(self, crossRatio : float):
        crossSpan = int(crossRatio * self.num_cities)
        assert(0<crossSpan<self.num_cities)
        children = np.zeros((self.cross_num, self.num_cities+1))
        for i in range(self.cross_num):
            parent_idx_i = np.random.choice(self.num_population, 2, replace=False)
            parent1 = self.population[parent_idx_i[0], 1:]
            parent2 = self.population[parent_idx_i[1], 1:]
            child = np.zeros_like(parent1)
            # cross over
            cross_start = np.random.choice(self.num_cities - crossSpan, 1)[0]
            cross_segment =  parent1[cross_start:(cross_start+crossSpan)]
            rest_segment = []
            for gene in parent2:
                if gene not in cross_segment:
                    rest_segment.append(gene)
            rest_segment = np.array(rest_segment)
            child = np.concatenate((rest_segment[:cross_start], cross_segment, rest_segment[cross_start:]))
            # update children
            children[i,0], children[i,1:] = self.calcFitness(child), child
        self.cross_children = children

    
    def mutate(self):
        mutation = np.zeros((self.mutate_num, self.num_cities+1))
        parent_idx = np.random.choice(self.num_population, self.mutate_num, replace=False)
        for i in range(self.mutate_num):
            swap_gene = np.random.choice(np.arange(1, self.num_cities), 2, replace=False)
            mutation[i] = self.population[parent_idx[i]]
            mutation[i,swap_gene[0]], mutation[i,swap_gene[1]] = mutation[i,swap_gene[1]], mutation[i,swap_gene[0]]
            mutation[i, 0] = self.calcFitness(mutation[i,1:])
        self.mutation_children = mutation
            
    
    def select(self):
        self.population = np.concatenate((self.population, self.cross_children, self.mutation_children), axis=0)
        select_idx = np.random.choice(self.num_population + self.cross_num + self.mutate_num, 
                                      self.num_population, replace=False,
                                      p = self.population[:,0]/np.sum(self.population[:,0]))
        self.population = self.population[select_idx]
        
    def reproduceGeneration(self) -> float:
        self.crossOver(0.4)
        self.mutate()
        self.select()
        return 10000.0 / self.population[:, 0].max()
    
    def getBestRoute(self) -> np.ndarray:
        max_idx = np.argmax(self.population[:, 0])
        return self.population[max_idx, 1:]
        
    def calcFitness(self, citiesOrder : np.ndarray) -> float:
        assert(citiesOrder.shape[0]==self.dist_mat.shape[0]==self.dist_mat.shape[1])
        idx1 = citiesOrder.astype(np.int64)
        idx2 = np.append(idx1[1:], idx1[0])
        dist = self.dist_mat[idx1, idx2]
        return 10000.0/np.sum(dist)    
  

def main(cities_location : np.ndarray):
    num_generation = 200
    dist_wrt_generations = []
    GA = generic()
    GA.initPopulation(cities_location)
    for i in range(num_generation):
        print("generation ", i)
        dist_wrt_generations.append(GA.reproduceGeneration())
    print("Best route:", GA.getBestRoute())
    
    
if __name__ == "__main__":
    cities_location=np.loadtxt('cities.txt')
    main(cities_location)