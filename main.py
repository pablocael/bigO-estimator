import numpy as np
import matplotlib.pyplot as plt
import time
import math
from timeit import default_timer as timer

minProcessTime = 10 # secs

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

def myFuncN2plusM2(v, w):
    result = []
    a = 1
    myFuncN2(v, w)
    myFuncN2(w, v)

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

def myFuncN_N(n):
    a = 1
    for i in range(0,n):

        a = a + i*i*i*i


def myFuncN_N2(n):
    a = 1
    for i in range(0,n):
        for i in range(0,n):
            a = a + i*i*i*i


def myFuncN_NLogN(n):
    a = 1
    v = dataFunctor(n)
    v.sort()


def myFuncNLogN(v, w):
    v.sort()

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

def measureFunc_N(func, n):
    start = time.process_time()
    func(n)
    end = time.process_time()
    return end - start

def measureAvg_N(func, n, numSamples):
    avg = 0
    for i in range(0, numSamples):
        avg = avg + measureFunc_N(func, n)
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

def measure_N(func, fr, to, inc):
    t = []
    x = []
    for i in range(fr, to, inc):
        x.append(i)

        avg = measureAvg_N(func, i, 1)
        t.append(avg)

    return (x, t)


def findMaxDatasetSize(func, dataFunc1, dataFunc2, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    datasetSize = 10
    avgTime = 0
    error = 0
    x = []
    t = []
    while avgTime < desiredWaitTime:
        avgTime = measureAvg(func, datasetSize, dataFunc1, dataFunc2, 2)
        if avgTime >= desiredWaitTime:
            break
        t.append(avgTime)
        x.append(datasetSize)
        if len(t) >= 10:
            # check if all previous 3 readings are the same
            if all(abs(p-t[0]) < 1e-4 for p in t[:5]):
                return (avgTime, 1, x, t, True)

        error = desiredWaitTime - avgTime
        datasetSize = int(datasetSize * (1+0.5*error))

    return (avgTime, datasetSize, x, t, False)

def findMaxNSize(func, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    Nsize = 10
    avgTime = 0
    error = 0
    x = []
    t = []
    while avgTime < desiredWaitTime:
        avgTime = measureAvg_N(func, Nsize, 2)
        if avgTime >= desiredWaitTime:
            break
        t.append(avgTime)
        x.append(Nsize)
        if len(t) >= 10:
            # check if all previous 3 readings are the same
            if all(abs(p-t[0]) < 1e-5 for p in t[:5]):
                return (avgTime, 1, x, t, True)

        error = desiredWaitTime - avgTime
        Nsize = int(Nsize * (1+0.5*error))

    return (avgTime, Nsize, x, t, False)

def printBigO(bigOFunc, varName, isConstant):
    if bigOFunc is not None:
        if isConstant:
            print('Function is in U', bigOFunc['label'])
        else:
            print('Function is in U', bigOFunc['label'].format('U'))
    else:
        print('Could not estimate U variable complexity ...')

def findBigO_UV(func, dataFunc1, dataFunc2):
    avgTime, datasetSizeU, x, t, isConstant = findMaxDatasetSize(func, dataFunc1, dataFunc2, minProcessTime)
    minError = 100000
    minFuncU = None
    if isConstant:
        minFuncU = functions[0]
    else:
        x = np.array(x)

        for f in functions:
            error = f['func'](x, t, datasetSizeU)
            if error < minError:
                minError = error
                minFuncU = f
        
    return (minFuncU, isConstant)

def findBigO_N(func):
    avgTime, Nsize, x, t, isConstant = findMaxNSize(func, minProcessTime)
    print('Nsize', Nsize)
    minError = 100000
    minFuncU = None
    if isConstant:
        minFuncU = functions[0]
    else:
        x = np.array(x)

        for f in functions:
            error = f['func'](x, t, Nsize)
            if error < minError:
                minError = error
                minFuncU = f

    
    result = 'N/A'
    if minFuncU is not None:
        if isConstant:
            result = 'O(1)'
        else:
           result = minFuncU['label'].format('N')
    return result

def findBigO(func):
    
    result = 'N/A'
    minFuncU, isConstant = findBigO_UV(func, dataFunctor, identityDataFunctor) 
    minFuncV, isConstant = findBigO_UV(func, identityDataFunctor, dataFunctor)    
   
    if minFuncU == None or minFuncV == None:
        return result

    minFunc, isConstant = findBigO_UV(func, dataFunctor, dataFunctor)  
 
    if minFunc is not None:
        if isConstant:
            result = 'O(1)'
        else:
            result = '{first} {{op}} {second}'.format(first=minFuncU['label'].format('U'), second=minFuncV['label'].format('V'))
            if minFunc == minFuncU or minFunc == minFuncV:
                result = result.format(op='+')
            else:
                result = result.format(op='*')

    return result


def test_UV(name, func, expected):
    start = time.time()
    print('*** starting test', name)
    result = findBigO(func)
    fail = result != expected
    print('FAIL' if fail else 'PASSED', ':', name, 'result:', result, 'expected', expected)
    print('*** proccess took', time.time() - start, 'seconds')

def test_N(name, func, expected):
    start = time.time()
    print('*** starting test', name)
    result = findBigO_N(func)
    fail = result != expected
    print('FAIL' if fail else 'PASSED', ':', name, 'result:', result, 'expected', expected)
    print('*** proccess took', time.time() - start, 'seconds')

test_N('test', myFuncN_NLogN, 'O(NLogN)')
test_N('test', myFuncN_N, 'O(N)')
test_N('test', myFuncN_N2, 'O(N^2)')

for i in range(0, 30):
    print('** running test sets', i)
    test_UV('N plus M', myFuncNplusM, 'O(U) + O(V)')
    test_UV('N2 plus M', myFuncN2plusM, 'O(U^2) + O(V)')
    test_UV('N2 plus M2', myFuncN2plusM2, 'O(U^2) + O(V^2)')
    test_UV('NLogN', myFuncNLogN, 'O(ULogU) + O(1)')
    test_UV('N', myFuncN, 'O(U) + O(1)')
    test_UV('N2', myFuncN2, 'O(U^2) + O(1)')
    test_UV('N3', myFuncN3, 'O(U) * O(V^2)')
    print('** end running test sets', i)

