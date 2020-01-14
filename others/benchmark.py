import numpy as np
import time as t

sizes = [10, 50, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000]

for size in sizes:
    A = np.random.random_sample((size, size))
    B = np.random.random_sample((size, size))
    start = t.time()
    C = A.dot(B)
    end = t.time()
    print("{} ---> {:.4f}".format(size, (end - start) * 1000))
