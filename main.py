import pandas as pd
import numpy as np
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

if __name__ == '__main__':
    model = '8x-5t-3t-2l'
    k = 10
    populationSize = 1
    data = pp.preprocess(pp.input('processedData.txt'))
    trainSet, testSet = pp.kFolds(data, k)

    print(trainSet.shape)
    print(testSet.shape)

    # for i in range(k):
    #     if i == 1:
    #         pso = PSO.ParticleSwarm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,:-2],trainSet[i][:,-2:]), model, populationSize)
    #         print(pso.testData[1][0])