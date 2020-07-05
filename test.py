import numpy as np
import math
import timeit

arr1 = np.array([1.23, 4.56])
arr2 = np.array([2.34, 5.67])
dif = arr2 - arr1

n = 100

def distance(arr1, arr2):
    return math.sqrt((arr1[0] - arr2[0]) ** 2 + (arr1[1] - arr2[1]) ** 2)

arr = np.vstack([dif] * n)
# vec_dif_time = timeit.timeit(lambda: np.sqrt(np.dot(arr, arr)))
dif_time = timeit.timeit(lambda: arr2 - arr1, number=n)
print('dif:', dif_time)
dot_sqrt = timeit.timeit(lambda: np.sqrt(np.dot(dif, dif)), number=n)
print('np dot-sqrt:', dot_sqrt)

# print(np.linalg.norm(arr, axis=1).shape)
linalg_norm = timeit.timeit(lambda: np.linalg.norm(arr, axis=1), number=1)
print('np linalg norm:', linalg_norm)

print(distance(arr1, arr2))
math_sqrt = timeit.timeit(lambda: distance(arr1, arr2), number=n)
print('math sqrt:', math_sqrt)

def rotationMatrix(cos, sin):
    return np.array([[cos, -sin], [sin, cos]])

vecRotationMatrix = np.vectorize(rotationMatrix)

a = np.random.rand(4, 2).astype(np.float64)
print(a)
print(a[:, 0])
print(a[:, 1])

thetas = np.arctan2(a[:, 1], a[:, 0])
print('theta:', thetas)
cos = np.cos(thetas)
sin = np.sin(thetas)
matricies = vecRotationMatrix(cos, sin)
print(matricies)
