import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocess as pp
import PSO
import MLP

def preprocessData(file='AirQualityUCI.xlsx'):
    data = pd.read_excel(file).values
    print(data.shape)
    data = data[:,[3,6,8,10,11,12,13,14,5]]
    print(data.shape)
    res = []
    for i in range(data.shape[0]):
        if -200.0 in data[i]:
            res.append(int(i))
    data = np.delete(data, res, 0)
    print(data.shape)
    pData = np.asarray([np.concatenate([data[l][:-1],[data[l+120][-1]],[data[l+240][-1]]]) for l in range(len(data)-240)])
    np.savetxt('processedData.txt', pData, fmt='%.13f')

def writeFile(fname, content):
    f = open(fname, 'w')
    f.write(content)
    f.close()

if __name__ == '__main__':
    model = '8x-3t-2l'
    act = MLP.modelInit(model)[2]
    k = 10
    epoch = 300
    populationSize = 30
    data = pp.preprocess(pp.input('processedData.txt'))
    trainSet, testSet = pp.kFolds(data, k)

    res1 = []
    res2 = []
    res3 = []

    for i in range(k):
        if i > -1:
            pso1 = PSO.ParticleSwarm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,:-2],trainSet[i][:,-2:]), model, populationSize)
            pso2 = PSO.ParticleSwarm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,:-2],trainSet[i][:,-2:]), model, populationSize)
            pso3 = PSO.ParticleSwarm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,:-2],trainSet[i][:,-2:]), model, populationSize)
            pso1.defineTopology([i for i in range(0)])
            pso2.defineTopology([i for i in range(2)])
            pso3.defineTopology([i for i in range(15)])
            

            agw1 = pso1.run(epoch)[0]
            agw2 = pso2.run(epoch)[0]
            agw3 = pso3.run(epoch)[0]
            oa1 = MLP.feedForward(testSet[i][:,:-2], agw1, 0, act)
            oa2 = MLP.feedForward(testSet[i][:,:-2], agw2, 0, act)
            oa3 = MLP.feedForward(testSet[i][:,:-2], agw3, 0, act)
            aac1 = MLP.getError(oa1[-1], testSet[i][:,-2:])
            aac2 = MLP.getError(oa2[-1], testSet[i][:,-2:])
            aac3 = MLP.getError(oa3[-1], testSet[i][:,-2:])
            res1.append(aac1)
            res2.append(aac2)
            res3.append(aac3)

            fig = plt.figure(i+1)
            plt.title('Fold '+str(i+1))
            plt.plot(np.arange(epoch), pso1.it, 'ro-', label='connectivity = 1', ms=5)
            plt.plot(np.arange(epoch), pso2.it, 'bo-', label='connectivity = 5', ms=5)
            plt.plot(np.arange(epoch), pso3.it, 'go-', label='connectivity = 30', ms=5)
            plt.legend(loc='best')
            fig.savefig('exp1,'+str((i+1))+'.png')
    
    writeFile('out1.txt', str(res1))
    writeFile('out2.txt', str(res2))
    writeFile('out3.txt', str(res3))



