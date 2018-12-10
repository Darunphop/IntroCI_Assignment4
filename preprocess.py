import numpy as np
from scipy import stats

def input(input):
    res = []
    with open(input, 'r') as inputFile:
        res = np.asarray([line.rstrip().split(',') for line in inputFile])
    return res

def preprocess(data):
    data = np.delete(data,0,1)
    data[:,0] = (data[:,0] == 'M').astype(float)
    data = data.astype(float)
    data[:,1:] = stats.zscore(data[:,1:],axis=0)
    return data

def kFolds(data, k=1):
    trainSet = [[] for i in range(k)]
    testSet = [[] for i in range(k)]
    dataSize = len(data)
    binSize = int(dataSize / k)
    remainSize = dataSize % k

    np.random.shuffle(data)

    for i in range(k):
        trainSet[i].extend(data[0:(i)*binSize])
        trainSet[i].extend(data[(i+1)*binSize:dataSize-remainSize])
        testSet[i].extend(data[i*binSize:(i+1)*binSize])
        if remainSize != 0:
            trainSet[i].extend(data[-(dataSize % k):])
    
    return np.asarray(trainSet), np.asarray(testSet)

if __name__ == 'preprocess':
    pass

if __name__ == '__main__':
    data = [[1,2,3,4],[4,5,6,7]]