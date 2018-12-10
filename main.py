import pandas as pd
import numpy as np

def preprocessData(file='AirQualityUCI.xlsx'):
    data = pd.read_excel(file).values
    print(data.shape)
    data = data[:,[3,6,8,10,11,12,13,14,5]]
    pData = np.asarray([np.concatenate([data[l][:-1],[data[l+120][-1]],[data[l+240][-1]]]) for l in range(len(data)-240)])
    np.savetxt('processedData.txt', pData, fmt='%.13f')

if __name__ == '__main__':
    pass