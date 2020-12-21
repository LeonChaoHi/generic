import numpy as np
import matplotlib.pyplot as plt


class generic():
    """Class of generic algorithm
    """
    def __init__(self) -> None:
        super().__init__()
        self.num_population = 100#随机生成的初始解的总数
        self.cross_num = 80#交叉解的个数
        self.mutate_num = 80#变异解的个数
        self.best_route = None
        
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
                                      p = np.exp(self.population[:,0])/np.sum(np.exp(self.population[:,0])))
        self.population = self.population[select_idx]
        
    def reproduceGeneration(self) -> float:
        self.crossOver(0.4)
        self.mutate()
        self.select()
        best_individual = self.population[np.argmax(self.population[:, 0])]
        if self.best_route is None or best_individual[0]>self.best_route[0]:
            self.best_route = best_individual
        return 10000.0 / best_individual[0]
    
    def getBestRoute(self):
        return 10000.0 / self.best_route[0], self.best_route[1:]
        
    def calcFitness(self, citiesOrder : np.ndarray) -> float:
        assert(citiesOrder.shape[0]==self.dist_mat.shape[0]==self.dist_mat.shape[1])
        idx1 = citiesOrder.astype(np.int64)
        idx2 = np.append(idx1[1:], idx1[0])
        dist = self.dist_mat[idx1, idx2]
        return 10000.0/np.sum(dist)    


def drawRoute(cities_location:np.ndarray, cities_route:np.ndarray) -> None:
    assert(cities_location.shape[1]==2)
    cities_route = np.append(cities_route, cities_route[0])
    cities_location = cities_location[cities_route.astype(np.int64), :]
    plt.plot(cities_location[:,0], cities_location[:,1], 'bo-')
    plt.show()


def main(cities_location : np.ndarray):
    num_generation = 500
    dist_wrt_generations = []
    GA = generic()
    GA.initPopulation(cities_location)
    for i in range(num_generation):
        print("generation ", i)
        dist_wrt_generations.append(GA.reproduceGeneration())
    print("Best route:", GA.getBestRoute())
    plt.plot(dist_wrt_generations)
    plt.show()
    _, best_route = GA.getBestRoute()
    drawRoute(cities_location, best_route)
    
    
if __name__ == "__main__":
    cities_location=np.loadtxt('cities.txt')
    main(cities_location)
