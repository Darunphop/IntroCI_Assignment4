import numpy as np
import MLP

class ParticleSwarm:
    class Node:
        def __init__(self, _id,  _w, _fitFunc, _act):
            self.neighbor = []
            self.w = _w
            self.speed = 0
            self.id = _id
            self.pBest = [0,0]
            self.fitFunc = _fitFunc
            self.act = _act
            self.fitness = 0
            self.timeStamp = 0

        def pairNeighbor(self, tNode):
            if tNode.id not in self.neighbor:
                self.neighbor.append(tNode)
                tNode.neighbor.append(self)

        def updateFitness(self, data, _timeStamp):
            if self.timeStamp < _timeStamp:
                o = self.fitFunc[1](data[0],self.w,0,self.act)
                self.fitness = 1.0 / self.fitFunc[0](o[-1],data[1])
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

if __name__ == '__main__':
    pass