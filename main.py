import numpy as np
import matplotlib.pyplot as plt
import time
import math
from timeit import default_timer as timer

minProcessTime = 6 # secs
numMeasureSamples = 6

def polyError(x, y, coeffs):
    poly =  np.zeros(len(x))
    for i in range(0, len(coeffs)):
        poly = poly + coeffs[coeffs.shape[0]-1 - i]*np.power(x, i)

    return np.sum((poly - y)**2)

def buildPolynomialData(x, y, maxDatasetSize, degree):
    A = np.zeros(len(x))
    for i in range(1, degree):
        A = np.c_[np.zeros(len(x)), A]
    
    A = np.c_[np.power(x, degree), A]
    print('A', A)
    
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    poly = coefficients[0]*np.power(x, degree)

    _ = plt.plot(x, y, 'o', label='Original data', markersize=3)
    _ = plt.plot(x, poly, label='Fitted line for N^' + str(degree))
    _ = plt.legend()

    return polyError(x, y, coefficients)

def buildN2Data(x, y, maxDatasetSize):
    return buildPolynomialData(x, y, maxDatasetSize, 2)

def buildN3Data(x, y, maxDatasetSize):
    return buildPolynomialData(x, y, maxDatasetSize, 3)

def buildNLogNData(x, y, maxDatasetSize):
    A = np.ones(len(x))
    A = np.c_[x*np.log(x), A]
    print('A', A)
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    curve = coefficients[0]*x*np.log(x+1) + coefficients[1]
    return np.sum((curve - y)**2)

def buildLinearData(x, y, maxDatasetSize):
    return buildPolynomialData(x, y, maxDatasetSize, 1)

def buildConstantData(x, y, maxDatasetSize):
    A = np.zeros(len(x))
    A = np.c_[np.ones(len(x)), A]

    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    poly =  np.zeros(len(x))
    for i in range(0, len(coefficients)):
        poly = poly + coefficients[coefficients.shape[0]-1 - i]*np.power(x, i)

    return polyError(x, y, coefficients)

functions = [
    {
        'func': buildConstantData,
        'label': 'O(1)'
    },
    {
        'func': buildLinearData,
        'label': 'O(N)'
    },
    {
        'func': buildNLogNData,
        'label': 'O(NLogN)'
    },
    {
        'func': buildN2Data,
        'label': 'O(N^2)'
    },
    {
        'func': buildN3Data,
        'label': 'O(N^3)'
    },
]

def myFuncN2(v):
    result = []
    a = 1
    for i in range(0,len(v)):
        for i in range(0,len(v)):
            a = a + i*i*i*i
            result.append(a)

    return result
    
def myFuncN(v):
    result = []
    a = 1
    for i in range(0,len(v)):

        a = a + i*i*i*i
        result.append(a)

    return result

def myFuncN3(v):
    result = []
    a = 1
    for i in range(0,len(v)):
        myFuncN2(v)

    return result


def myFuncNLogN(v):
    result = v.tolist()
    result.sort()

def measureFunc(func, v):
    start = time.process_time()
    func(v)
    end = time.process_time()
    return end - start

def measureAvg(func, dataSetSize, dataFunc, numSamples):
    avg = 0
    for i in range(0, numSamples):
        avg = avg + measureFunc(func, dataFunc(dataSetSize))
    return avg / numSamples

def dataFunctor(size):
    return np.random.randint(low = 0, high = 1000, size = size)

def measure(func, fr, to, inc):
    t = []
    x = []
    for i in range(fr, to, inc):
        print('..', i)
        x.append(i)

        avg = measureAvg(func, i, dataFunctor, 1)
        t.append(avg)

    return (x, t)

def findMaxDatasetSize(func, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    datasetSize = 100
    avgTime = 0
    while avgTime < desiredWaitTime:
        avgTime = measureAvg(func, datasetSize, dataFunctor, 2)
        if avgTime >= desiredWaitTime:
            break
        error = desiredWaitTime - avgTime
        datasetSize = int(datasetSize * (1 + 0.35 * error))
    
    return (avgTime, datasetSize)


def findBigO(func):
    avgTime, datasetSize = findMaxDatasetSize(func, minProcessTime)

    x, t = measure(func, 1, datasetSize, int(datasetSize / numMeasureSamples))
    x = np.array(x)

    minError = 100000
    minFunc = None

    for f in functions:
        error = f['func'](x, t, datasetSize)
        if error < minError:
            minError = error
            minFunc = f
        
    if minFunc is not None:
        print('Function is ', minFunc['label'], 'with error', minError)

findBigO(myFuncN3)

plt.show()