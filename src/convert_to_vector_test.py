import numpy as np
import loader

x = 2
arr1 = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
arr2 = np.zeros(10)
arr2[x] = 1

print(arr1)
print(arr2)

ld = loader.loader()
data = ld.load_data("data/mnist_test.csv")
print(data[0][0])
labels = data[:,0]
data_size = data.shape[0]
data[:,0] = np.zeros((data_size, 10))
print(labels)
print(data[:,0])