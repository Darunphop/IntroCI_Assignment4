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
                self.neighbor.append(tNode.id)
                tNode.neighbor.append(self.id)

        def updateFitness(self, data, _timeStamp):
            if self.timeStamp < _timeStamp:
                o = self.fitFunc[1](data[0],self.w,0,self.act)
                self.fitness = self.fitFunc[0](o[-1],data[1])
                self.timeStamp = _timeStamp

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

if __name__ == '__main__':
    pass