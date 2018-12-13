import activation as act
import numpy as np

def modelInit(model):
    layerSize = [int(n[:-1]) for n in model.split('-')]
    activationLayer = [n[-1:] for n in model.split('-')]
    nHidden = len(layerSize) - 1

    weight = []
    for i in range(nHidden):
        weight.append(np.random.randn(layerSize[i+1], layerSize[i]))
        # weight.append(np.ones((layerSize[i+1], layerSize[i])) / 5**-2)
    bias = []
    for i in range(nHidden):
        bias.append(np.random.randn(layerSize[i+1]))

    return weight, bias, activationLayer

def feedForward(input, weigth, bias, activation):
    res = []
    tmp = input
    

    for i in range(len(activation)-1):
        # interm = 0
        # if len(tmp)==0:
        #     interm = input
        # else:
        #     interm = tmp
        tmp = np.dot(tmp, np.transpose(weigth[i]))
        # for j in range(tmp.shape[0]):
        #     for k in range(tmp.shape[1]):
        #         tmp[j][k] += bias[i][k]
        tmp = act.activate(np.copy(tmp), activation[i+1])
        res.append(np.asarray(tmp))
    return res

def classInterprete(y):
    return (y[:,0] <= y[:,1]).astype(int)

def getError(y, d):

    diff = abs(d - y)

    return np.average(np.sum(diff,axis=1))