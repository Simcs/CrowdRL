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

print()

def rot_mat(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])

pos = np.array([ 9.8219547 , -0.02838143])
vel = np.array([-0.0901949 , -0.00434144])
force = np.array([0.09502771 , 0.00470591]).reshape(2, 1)
dt = 1e-1

theta = np.arctan2(vel[1], vel[0])
ori = rot_mat(theta)
print('theta:', theta)
print('ori:', ori)

new_pos = pos + dt * vel
new_vel = vel + dt * (ori @ force).reshape(2)

print('new pos:', new_pos)
print('new vel:', new_vel)

new_theta = np.arctan2(new_vel[1], new_vel[0])
new_ori = rot_mat(new_theta)

print('new theta:', new_theta)
print('new ori:', new_ori)


# pos: [ 9.8219547  -0.02838143]
# vel: [-0.0901949  -0.00434144]
# theta: -3.0934957589308647
# ori: [[-0.99884357  0.04807835]
#  [-0.04807835 -0.99884357]]
# force: [0.09502771 0.00470591]
# new_pos: [ 9.81293521 -0.02881557]
# new_vel: [-0.09966405 -0.00526837]
# new_theta: -3.088780547747382
# new_ori: [[-0.99860576  0.05278756]
#  [-0.05278756 -0.99860576]]

n = 5
a = np.array([np.pi, np.pi / 2, 3])
dw = np.array([(i*np.pi) / (n-1) - np.pi/2 for i in range(n)])
w = np.array([a1 + dw for a1 in a])
print(w.shape)
cos = np.cos(w)
sin = np.sin(w)
d = np.empty((3, n, 2))
d[:, :, 0] = np.cos(w)
d[:, :, 1] = np.sin(w)
print(d)

print(d[0][0])
print(d[0][n-1])
print(d[1][0])
print(d[1][1])

a = np.array([[0, 1]])

b = np.array([[1, 2], [2, 3]])

print(np.concatenate((a, b), axis=0))