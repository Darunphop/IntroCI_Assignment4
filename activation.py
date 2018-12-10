import numpy as np

def sigmoidO(x):
    return 1.0 / (1 + np.exp(0.0-x))

def inverseSigmoid(x):
    return sigmoidO(x) * (1.0 - sigmoidO(x))

sigmoid = np.vectorize(sigmoidO)

def activate(x, func, div=False):
    res = x
    if func == 's':
        if div:
            res = inverseSigmoid(x)
        else:
            res = sigmoid(x)
    if func == 't':
        if div:
            res = 1.0 - np.tanh(x)**2
        else:
            res = np.tanh(x)
    if func == 'l':
        if div:
            res = 1. * (x > 0)
        else:
            res = np.maximum(x, 0)
    return res

if __name__ == '__main__':
    sm = np.vectorize(sigmoid)
    print(sm([1,2,3]))