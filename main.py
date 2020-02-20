import numpy as np
import matplotlib.pyplot as plt
import time

def buildPolynomialData(x, degree):
    pass

def buildLogNData(x):
    pass

def buildLinearData(x):
    pass

functions = [{
    'func': lambda x: np.power(x, 2),
    'label': 'O(N^2)'
}]

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 4, 9])
print(x)
A = np.vstack([np.power(x,2)]).T
print(A, A.shape)

print(np.power(A[:,0],2))
A = np.c_[A, x]
A = np.c_[A, np.ones(A.shape[0])]
print(A, A.shape)

m, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, b, c)


_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x*x + b*x + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()

def myFunction(V):
    pass



