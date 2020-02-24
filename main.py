import numpy as np
import matplotlib.pyplot as plt
import time
import math
from timeit import default_timer as timer

minProcessTime = 3 # secs
numMeasureSamples = 10


def polyError(x, y, coeffs):
    poly =  np.zeros(len(x))
    for i in range(0, len(coeffs)):
        poly = poly + coeffs[coeffs.shape[0]-1 - i]*np.power(x, i)

    return np.sum((poly - y)**2)

def buildPolynomialData(func, maxDatasetSize, degree):

    x, t = measure(func, 0, maxDatasetSize, int(maxDatasetSize / numMeasureSamples))
    x = np.array(x)
    print(x, t)
    A = np.ones(len(x))
    A = np.c_[x, A]
    for i in range(2, degree+1):
        A = np.c_[np.power(x, i), A]
        print(A, A.shape)
 
    coefficients = np.linalg.lstsq(A, t, rcond=None)[0]
    poly =  np.zeros(len(x))
    for i in range(0, len(coefficients)):
        poly = poly + coefficients[coefficients.shape[0]-1 - i]*np.power(x, i)

    e = polyError(x, t, coefficients)
    print('total error^2 = ', e)

    _ = plt.plot(x, t, 'o', label='Original data', markersize=3)
    _ = plt.plot(x, poly, 'r', label='Fitted line')
    _ = plt.legend()

def buildN2Data(func, maxDatasetSize):
    return buildPolynomialData(func, 2)

def buildN3Data(func, maxDatasetSize):
    return buildPolynomialData(func, 3)

def buildNLogNData(func, maxDatasetSize):
    x, t = measure(func, 1, maxDatasetSize, int(maxDatasetSize / numMeasureSamples))
    x = np.array(x)
    print(x, t)
    A = np.ones(len(x))

    A = np.c_[x*np.log(x), A]
    print(A, A.shape)
 
    coefficients = np.linalg.lstsq(A, t, rcond=None)[0]
    print('>>>> COEFFS', coefficients)
    curve = coefficients[0]*x*np.log(x) + coefficients[1]
    print('>>> CURVE = ', curve)
    e = np.sum((curve - t)**2)
    print('total error^2 = ', e)

    _ = plt.plot(x, t, 'o', label='Original data', markersize=3)
    _ = plt.plot(x, curve, 'r', label='Fitted line')
    _ = plt.legend()

def buildLinearData(func, maxDatasetSize):
    pass

functions = [
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

def partition(arr,low,high): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[high]     # pivot 
  
    for j in range(low , high): 
  
        # If current element is smaller than or 
        # equal to pivot 
        if   arr[j] <= pivot: 
          
            # increment index of smaller element 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 
  
# Function to do Quick sort 
def quickSort(arr,low,high): 
    if low < high: 
  
        # pi is partitioning index, arr[p] is now 
        # at right place 
        pi = partition(arr,low,high) 
  
        # Separately sort elements before 
        # partition and after partition 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 

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

        avg = measureAvg(func, i, dataFunctor, 5)
        t.append(avg)

    return (x, t)

def findMaxDatasetSize(func, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    datasetSize = 1000
    avgTime = 0
    while avgTime < desiredWaitTime:
        avgTime = measureAvg(func, datasetSize, dataFunctor, 2)
        if avgTime >= desiredWaitTime:
            break
        error = desiredWaitTime - avgTime
        datasetSize = int(datasetSize * (1 + 0.35 * error))
    
    return (avgTime, datasetSize)


#measure(myFuncN2, 0, 1000, 100)

#measure(myFuncNLogN, 0, 15000, 1000)

#measure(myFuncN, 0, 150000, 10000)
#plt.show()

avgTime, datasetSize = findMaxDatasetSize(myFuncN, minProcessTime)
print('found dataset size = ', avgTime, datasetSize)
buildNLogNData(myFuncN, datasetSize)
buildPolynomialData(myFuncN, datasetSize, 1)
buildPolynomialData(myFuncN, datasetSize, 2)

plt.show()