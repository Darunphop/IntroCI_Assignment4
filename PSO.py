import numpy as np

class ParticleSwarm:
    class Node:
        def __init__(self, _id,  _w, _fitFunc):
            self.neighbor = []
            self.w = _w
            self.speed = 0
            self.id = _id
            self.pBest = [0,0]
            self.fitFunc = _fitFunc
            self.fitness = 0
            self.timeStamp = 0

        def pairNeighbor(self, tNode):
            if tNode.id not in self.neighbor:
                self.neighbor.append(tNode.id)
                tNode.neighbor.append(self.id)

        def updateFitness(self, data, _act, _timeStamp):
            if self.timeStamp < _timeStamp:
                o = self.fitFunc[1](data[0],self.w,0,_act)
                self.fitness = self.fitFunc[0](o[-1],data[1])
                self.timeStamp = _timeStamp

    def __init__(self):
        pass

if __name__ == '__main__':
    pass