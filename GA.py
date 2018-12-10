import numpy as np
import copy

class GeneticAlgorithm:
    def __init__(self, fitnessFunc, initFunc, testData, seed, nPopulation, sRate, mRate):
        self.population = []
        self.fitnessFunc = fitnessFunc
        self.initFunc = initFunc
        self.testData = testData
        self.model = seed
        self.nPopulation = nPopulation
        self.sRate = sRate
        self.mRate = mRate
        self.gBest = [0,0]
        self.it = 0

        self.fitness = np.zeros(nPopulation)
        self.act = self.initFunc(self.model)[2]

        for i in range(nPopulation):
            self.population.append(self.initFunc(self.model)[0])

    def run(self, it):
        res = []
        for i in range(it):
            print('iteration ', i)
            self.updateFitness()
            self.nextPopulation()
            self.updateFitness()
            self.updateGBest()
            res.append(self.getFitest()[0])
        return res


    def updateFitness(self, data=0):
        if data == 0:
            data = self.testData 
        self.fitness = np.zeros(self.nPopulation)
        for i in range(self.nPopulation):
            o = self.fitnessFunc[1](data[0],self.population[i],0,self.act)
            self.fitness[i] = self.fitnessFunc[0](o[-1],data[1])

    def selection(self):
        csum = np.cumsum(self.fitness)
        spareSize = (int(self.nPopulation * self.sRate) >> 1) << 1

        id = []
        for i in range(spareSize):
            x = csum[-1] * np.random.ranf()
            id.append(np.argmax(csum>x))

        itmPopuplation = np.asarray(self.population)[id]
        self.population = np.delete(self.population, id, axis=0).tolist()
        self.fitness = np.delete(self.fitness, id)

        # sum1 = 0
        # for i in self.population:
        #     x = np.concatenate((i[0],i[1],i[2]), axis=None)
        #     sum1 += x.sum()
        # print('after sum1', sum1)
        drop = spareSize - len(list(set(id)))
        # print('drop', drop)
        if drop > 0:
            rIdx = np.argpartition(self.fitness, drop)
            # print(rIdx[:drop])
            self.population = np.delete(self.population, rIdx[:drop], axis=0).tolist()
            self.fitness = np.delete(self.fitness, rIdx)
        
        # matedPop = self.mating(itmPopuplation)
        # self.mutate(matedPop)

        return itmPopuplation

    def nextPopulation(self):
        itmPop = self.selection()
        newPop = np.concatenate((self.population, itmPop))
        # print(self.hash(newPop))
        self.mutate(newPop)
        # print(self.hash(newPop))
        self.population = newPop.tolist()
        self.it += 1

    def mating(self, population):
        res = []
        np.random.shuffle(population)
        for i in range(int(population.shape[0]/2)):
            if i != -1:
                res.extend(self.crossover(population[i], population[i+1]))
        return np.asarray(res)

    def crossover(self, i1, i2):
        shape = [(i.shape[0],i.shape[1]) for i in i1]
        size = np.cumsum([i.shape[0]*i.shape[1] for i in i1])[:-1]
        chromosome1 = np.concatenate([i for i in i1], axis=None)
        chromosome2 = np.concatenate([i for i in i2], axis=None)

        cp = [0,0]

        while abs(cp[0] - cp[1]) < 0.10*chromosome1.shape[0]: 
            cp = np.random.randint(0,chromosome1.shape[0],2)
        cp.sort()

        tmp = chromosome1[cp[0]:cp[1]].copy()
        chromosome1[cp[0]:cp[1]] = chromosome2[cp[0]:cp[1]].copy()
        chromosome2[cp[0]:cp[1]] = tmp

        w1 = [i.reshape(shape[it]) for it,i in enumerate(np.split(chromosome1, size))]
        w2 = [i.reshape(shape[it]) for it,i in enumerate(np.split(chromosome2, size))]

        return [w1,w2]

    def mutate(self, population):
        size = 1000
        norm = np.random.normal(0,0.6,size)
        # print(norm.max(), norm.min())
        mask = np.random.choice([0, 1], size=(size,), p=[1.-self.mRate, self.mRate])
        # print(norm)
        # print(mask)
        # print(np.multiply(norm, mask))


        for it in range(len(population)):
            c,sh,si = self.toChromosome(population[it])
            norm = np.random.normal(0,0.6,c.shape[0])
            mask = np.random.choice([0, 1], size=(c.shape[0],), p=[1.-self.mRate, self.mRate])
            tmpC = np.multiply(norm, mask)
            c += tmpC
            population[it] = self.toValue(c,sh,si)

    def getFitest(self):
        return np.amax(self.fitness), np.argmax(self.fitness)

    def updateGBest(self):
        lBest = self.getFitest()
        if lBest[0] > self.gBest[0]:
            self.gBest = [lBest[0], self.population[lBest[1]]]

    def toChromosome(self, value):
        shape = [(i.shape[0],i.shape[1]) for i in value]
        size = np.cumsum([i.shape[0]*i.shape[1] for i in value])[:-1]
        chromosome = np.concatenate([i for i in value], axis=None)

        return chromosome, shape, size

    def toValue(self, chromosome, shape, size):
        return [i.reshape(shape[it]) for it,i in enumerate(np.split(chromosome, size))]

    def hash(self, w):
        sum = 0
        for i in w:
            sum += np.asarray([j.sum() for j in i]).sum()
        return sum
    
    def showStructure(self, w):
        for i in w:
            print([j.shape for j in i])