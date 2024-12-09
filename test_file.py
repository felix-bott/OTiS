import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
import time

print("you are here1")

testArray = np.array([0,1,2,3,4,5,4,2,1,3])

# plt.plot(testArray)
# plt.savefig('/vol/aimspace/users/bofe/TurBo/OTiS/test_plot.png')

print("you are here2")
for i in range(0,50):
    print(i)
    t0 = time.time()
    A = np.random.rand(5000, 5000)
    SPD_matrix = np.dot(A, A.T)
    SPD_matrix_inverse = np.linalg.inv(SPD_matrix)
    t1 = time.time()
    print(t1-t0)

