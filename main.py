import numpy as np
import matplotlib.pyplot as plt
import time
import math
from timeit import default_timer as timer

minProcessTime = 6 # secs
numMeasureSamples = 8

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
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return polyError(x, y, coefficients)

def buildN2Data(x, y, maxDatasetSize):
    return buildPolynomialData(x, y, maxDatasetSize, 2)

def buildN3Data(x, y, maxDatasetSize):
    return buildPolynomialData(x, y, maxDatasetSize, 3)

def buildNLogNData(x, y, maxDatasetSize):
    A = np.ones(len(x))
    A = np.c_[x*np.log(x), A]
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
        'label': 'O({0})'
    },
    {
        'func': buildNLogNData,
        'label': 'O({0}Log{0})'
    },
    {
        'func': buildN2Data,
        'label': 'O({0}^2)'
    },
    {
        'func': buildN3Data,
        'label': 'O({0}^3)'
    },
]

def myFuncN2plusM(v, w):
    result = []
    a = 1
    myFuncN2(v, w)
    myFuncN(w, v)

    return result

def myFuncNplusM(v, w):
    result = []
    a = 1
    for i in range(0,len(v)):
            a = a + i*i*i*i
            result.append(a)

    for i in range(0,len(w)):
            a = a + i*i*i*i
            result.append(a)

    return result

def myFuncN2(v, w):
    result = []
    a = 1
    for i in range(0,len(v)):
        for i in range(0,len(v)):
            a = a + i*i*i*i
            result.append(a)

    return result
    
def myFuncN(v, w):
    result = []
    a = 1
    for i in range(0,len(v)):

        a = a + i*i*i*i
        result.append(a)

    return result

def myFuncN3(v, w):
    result = []
    a = 1
    for i in range(0,len(v)):
        myFuncN2(w, v)

    return result

def myFuncNLogN(v, w):
    result = v.tolist()
    result.sort()

def measureFunc(func, v, w):
    start = time.process_time()
    func(v, w)
    end = time.process_time()
    return end - start

def measureAvg(func, dataSetSize, dataFunc1, dataFunc2, numSamples):
    avg = 0
    for i in range(0, numSamples):
        avg = avg + measureFunc(func, dataFunc1(dataSetSize), dataFunc2(dataSetSize))
    return avg / numSamples

def dataFunctor(size):
    return np.random.randint(low = 0, high = 1000, size = size)

def identityDataFunctor(size):
    return [0]

def measure(func, dataFunc1, dataFunc2, fr, to, inc):
    t = []
    x = []
    for i in range(fr, to, inc):
        print('..', i)
        x.append(i)

        avg = measureAvg(func, i, dataFunc1, dataFunc2, 1)
        t.append(avg)

    return (x, t)

def findMaxDatasetSize(func, dataFunc1, dataFunc2, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    datasetSize = 100
    avgTime = 0
    prev3 = []
    while avgTime < desiredWaitTime:
        avgTime = measureAvg(func, datasetSize, dataFunc1, dataFunc2, 2)
        if avgTime >= desiredWaitTime:
            break
        prev3.append(avgTime)
        if len(prev3) == 3:
            if all(abs(p-prev3[0]) < 1e-5 for p in prev3):
                return (avgTime, 1, True)
        error = desiredWaitTime - avgTime
        datasetSize = int(datasetSize * (1 + 0.35 * error))
    
    return (avgTime, datasetSize, False)

def printBigO(bigOFunc, varName, isConstant):
    if bigOFunc is not None:
        if isConstant:
            print('Function is in U', bigOFunc['label'])
        else:
            print('Function is in U', bigOFunc['label'].format('U'))
    else:
        print('Could not estimate U variable complexity ...')

def findBigO_UV(func, dataFunc1, dataFunc2):
    print('estimating good dataset size ...')
    avgTime, datasetSizeU, isConstant = findMaxDatasetSize(func, dataFunc1, dataFunc2, minProcessTime)
    print('found dataset size of', datasetSizeU)
    minError = 100000
    minFuncU = None
    x, t = [], []
    if isConstant:
        minFuncU = functions[0]
    else:
        x, t = measure(func, dataFunc1, dataFunc2, 1, datasetSizeU, int(datasetSizeU / numMeasureSamples))
        x = np.array(x)

        for f in functions:
            error = f['func'](x, t, datasetSizeU)
            if error < minError:
                minError = error
                minFuncU = f
        
    return (minFuncU, isConstant)

def findBigO(func):
    
    minFuncU, isConstant = findBigO_UV(func, dataFunctor, identityDataFunctor) 
    if minFuncU is not None:
        if isConstant:
            print('Function is in U,', minFuncU['label'])
        else:
            print('Function is in U,', minFuncU['label'].format('U'))
    else:
        print('Could not estimate U variable complexity ...')

    minFuncV, isConstant = findBigO_UV(func, identityDataFunctor, dataFunctor)    
    if minFuncV is not None:
        if isConstant:
            print('Function is in V,', minFuncV['label'])
        else:
            print('Function is in V,', minFuncV['label'].format('V'))
    else:
        print('Could not estimate V variable complexity ...')

    if minFuncU == None or minFuncV == None:
        return

    minFunc, isConstant = findBigO_UV(func, dataFunctor, dataFunctor)    
    if minFunc is not None:
        if isConstant:
            print('Function does not depend on neight U nor V')
        else:
            if minFunc == minFuncU or minFuncU == minFuncV:
                print('Function is combined U and V', minFuncU['label'].format('U'), '+', minFuncV['label'].format('V'))
            else:
                print('Function is combined U and V', minFuncU['label'].format('U'), '*', minFuncV['label'].format('V'))

start = time.time()
findBigO(myFuncN2plusM)
print('>>> proccess took', time.time() - start, 'seconds')
