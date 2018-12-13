import numpy as np
import MLP

class ParticleSwarm:
    class Node:
        def __init__(self, _id,  _w, _fitFunc, _act):
            self.neighbor = [self]
            self.w = _w
            self.speed = 0
            self.id = _id
            self.pBest = [0,0]
            self.fitFunc = _fitFunc
            self.act = _act
            self.fitness = 0
            self.timeStamp = -1
            self.v = []

            for i in range(len(_w)):
                self.v.append(np.zeros(_w[i].shape))

        def pairNeighbor(self, tNode):
            if tNode not in self.neighbor and tNode is not self:
                self.neighbor.append(tNode)
                tNode.neighbor.append(self)

        def updateFitness(self, data, _timeStamp):
            if self.timeStamp < _timeStamp:
                o = self.fitFunc[1](data[0],self.w,0,self.act)
                fitness = 1.0 / self.fitFunc[0](o[-1],data[1])
                if fitness > self.pBest[0]:
                    self.pBest = [fitness, self.w]
                self.fitness = fitness
                self.timeStamp = _timeStamp

        def getWDiff(self, _w):
            res = []
            for i in range(len(self.w)):
                tmp = _w[i] - self.w[i]
                res.append(tmp)
            return res

        def getLBest(self):
            lbestf = 0
            lbestn = []
            for i in self.neighbor:
                if i.fitness > lbestf:
                    lbestf = i.fitness
                    lbestn = i.w
            return [lbestf, lbestn]

        def updateVelocity(self, c1=1, c2=1):
            r1 = np.random.rand()
            r2 = np.random.rand()
            cognitive = [(r1*c1)*l for l in self.getWDiff(self.pBest[1])]
            social = [(r2*c2)*(l) for l in self.getWDiff(self.getLBest()[1])]
            self.v = [self.v[i] + cognitive[i] + social[i] for i in range(len(self.v))]

        def updatePosition(self):
            newW = [self.w[i] + self.v[i] for i in range(len(self.w))]
            self.w = newW

    def __init__(self, fitnessFunc, initFunc, testData, seed, nPopulation):
        self.population = []
        self.fitnessFunc = fitnessFunc
        self.initFunc = initFunc
        self.testData = testData
        self.model = seed
        self.nPopulation = nPopulation
        self.gBest = [0,0]
        self.it = 0

        for i in range(nPopulation):
            w,_,act = MLP.modelInit(self.model)
            n = self.Node(i,w,fitnessFunc,act)
            self.population.append(n)

    def defineTopology(self, connections):
        for i in range(self.nPopulation):
            for j in connections:
                target = (i + j) % self.nPopulation
                self.population[i].pairNeighbor(self.population[target])

    def run(self, maxIt):

        for i in range(maxIt):
            self.updateFitness(i)
            
            iGbest = self.getGBest()
            if self.gBest[0] < iGbest[0]:
                self.gBest = iGbest
            

            self.updateVelocity()
            self.updatePosition()

            print(i, self.gBest[0], iGbest[0])


    def updateFitness(self, _it):
        for n in self.population:
            n.updateFitness(self.testData, _it)
    
    def updateVelocity(self):
        for n in self.population:
            n.updateVelocity()

    def updatePosition(self):
        for n in self.population:
            n.updatePosition()

    def getGBest(self, allTime=True):
        gbestv = 0
        gbestw = []
        for n in self.population:
            if n.fitness > gbestv:
                gbestv = n.fitness
                gbestw = n.w
        return [gbestv, gbestw]


if __name__ == '__main__':
    pass