import numpy as np
import matplotlib.pyplot as plt
import time
import math
from timeit import default_timer as timer

# This function takes last element as pivot, places 
# the pivot element at its correct position in sorted 
# array, and places all smaller (smaller than pivot) 
# to left of pivot and all greater elements to right 
# of pivot 
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

# def buildPolynomialData(x, degree):
#     pass

# def buildN2Data(x):
#     return buildPolynomialData(x, 2)

# def buildN3Data(x):
#     return buildPolynomialData(x, 3)

# def buildN4Data(x):
#     return buildPolynomialData(x, 4)

# def buildLogNData(x):
#     pass

# def buildLinearData(x):
#     pass

# functions = [{
#     'func': buildPolynomialData,
#     'label': 'O(N^2)'
# }]

# x = np.array([0, 2, 3, 4, 5, 6])
# y = np.array([0, x[1]**3, x[2]**3, x[3]**3, x[4]**3, x[5]**3])
# print(x)
# A = np.vstack([np.power(x,2)]).T
# print(A, A.shape)

# print(np.power(A[:,0],2))
# A = np.c_[A, x]
# A = np.c_[A, np.ones(A.shape[0])]
# print(A, A.shape)

# m, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
# print(m, b, c)

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
    quickSort(result, 0, len(result)-1)

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
        print('measure', i)
        x.append(i)

        avg = measureAvg(func, i, dataFunctor, 20)
        t.append(avg)

    print('finished measuring', x, t)
    _ = plt.plot(x, t, 'o', label='OMeasure', markersize=2)
# _ = plt.plot(x, x*x, 'r', label='Fitted line N2')
# _ = plt.plot(x, x*x*x, 'g', label='Fitted line N3')
# _ = plt.legend()

def findMaxDatasetSize(func, desiredWaitTime):
    # determines the dataset size for keeping process runtime at about 10 seconds
    datasetSize = 1000
    avgTime = 0
    while avgTime < desiredWaitTime:
        print('trying size = ', datasetSize)
        avgTime = measureAvg(func, datasetSize, dataFunctor, 2)
        print('found time = ', avgTime, 'for size = ', datasetSize)
        if avgTime >= desiredWaitTime:
            break
        error = desiredWaitTime - avgTime
        datasetSize = int(datasetSize * (1 + 0.25 * error))
    
    return (avgTime, datasetSize)


#measure(myFuncN2, 0, 1000, 100)

#measure(myFuncNLogN, 0, 15000, 1000)

#measure(myFuncN, 0, 150000, 10000)
#plt.show()

print('data set for N2', findMaxDatasetSize(myFuncN2, 4))