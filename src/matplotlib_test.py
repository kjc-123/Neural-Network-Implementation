import loader
import random

import numpy as np
#import matplotlib.pyplot
import matplotlib.pyplot as plt

ld = loader.loader()

X_train, y_train = ld.load_data("data/mnist_train.csv")

nrows = 5
ncols = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

for i in range(0,nrows):
    for j in range(0,ncols):
        index = random.randint(0, 60000)
        image = X_train[index]
        image = np.reshape(image, shape=(28, 28))
        axes[i,j].imshow(image, cmap='gray')
        predicted = str(np.where(y_train[index] == 1.0)[0][0]) # TODO: put model prediction here
        title = "Predicted: " + predicted + " | Actual: " + str(np.where(y_train[index] == 1.0)[0][0])
        axes[i,j].set_title(title, fontsize=8)

'''for i in range(0, 5):
    index = random.randint(0, 60000)
    image = X_train[index]
    image = np.reshape(image, shape=(28, 28))
    plt.imshow(image, cmap='gray')
    predicted = str(np.where(y_train[index] == 1.0)[0][0]) # TODO: put model prediction here
    title = "Predicted: " + predicted + " | Actual: " + str(np.where(y_train[index] == 1.0)[0][0])
    plt.title(title)'''

plt.tight_layout()
plt.show()

'''train_input, train_output = ld.load_data("data/mnist_train.csv")
test_input, test_output = ld.load_data("data/mnist_test.csv")

print(train_input[0])
print(train_output[0])'''

#training_input = training_data[:, 1:785]
#testing_input = testing_data[:, 1:785]

#training_output = [loader.loader.vectorized_result(y) for y in training_data[:, 0]]
#testing_output = testing_data[:, 0]
#print(testing_input[0])
#print(training_output[0])
#print([y for y in int(training_data[:,0])])

'''training_output = [int(i) for i in training_data[:,0]]
training_output = [ld.vectorized_result(y) for y in training_output]
testing_output = [int(i) for i in testing_data[:,0]]
testing_output = [ld.vectorized_result(y) for y in testing_output]
print(training_output)
'''

'''training_output = training_data[:,0]
testing_output = testing_data[:,0]
print(training_output)'''