import numpy as np
import loader

# TODO: should sigmoid be in forward prop? should initial values be N(0,1) or U(0,1)?

class MyNetwork():

    # used 3blue1brown's neural network series for initial intuition
    # used ChatGPT for deeperexplanation of math
    # used Claude AI for some debugging
     
    # Step 1: initialize weights and biases
    # TODO: generalize size and number of hidden layers
    # TODO: fix dimensions (they're all backwards lol)
    # TODO: take code out of sgd() and put into other functions
    # He initialization
    W1 = np.random.randn(784, 16) * np.sqrt(2.0 / 784) # randn is standard normal distribution Z ~ N(0, 1)
    W2 = np.random.randn(16, 10) * np.sqrt(2.0 / 16)
    b1 = np.zeros([1, 16])
    b2 = np.zeros([1, 10])

    ld = loader.loader()

    X_train, y_train = ld.load_data("data/mnist_train.csv")
    X_test, y_test = ld.load_data("data/mnist_test.csv")

    # normalize data (very important!!!)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    def __init__(self):
        print("network initialized")
    
    def sgd(self):

        eta = 1.0 # learning rate
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1
        b2 = self.b2
        r = 600
        m = int(60000 / r)

        for i in range(0, r): # r epochs

            W1 = self.W1
            W2 = self.W2
            b1 = self.b1
            b2 = self.b2

            # Step 2a: sample minibatch of size m from training data
            i0 = (m * i)
            il = (m * (i+1))
            X_batch = self.X_train[i0:il,:]
            y_batch = self.y_train[i0:il]

            # Step 2b: forward pass
            Z1 = self.forwardprop(X_batch, W1, np.ones([m,1]) @ b1)
            A1 = self.relu(Z1)
            Z2 = self.forwardprop(A1, W2, np.ones([m,1]) @ b2)
            A2 = self.sigmoid(Z2)

            # Step 2c: backward pass
            del2 = np.multiply(A2 - y_batch, np.multiply(A2, np.ones(A2.shape)-A2))
            # print(del2.shape)
            del1 = np.multiply(del2 @ np.transpose(W2), self.relu_derivative(Z1))

            # Step 2d: calculate average gradients over batch
            delW2 = (np.transpose(A1) @ del2) / m
            delb2 = np.ones([1, m]) @ del2 / m
            delW1 = (np.transpose(X_batch) @ del1) / m
            delb1 = np.ones([1, m]) @ del1 / m

            # Step 2e: update parameters
            #print(del3[0])
            self.W1 = self.W1 - (eta * delW1)
            self.b1 = self.b1 - (eta * delb1)
            self.W2 = self.W2 - (eta * delW2)
            self.b2 = self.b2 - (eta * delb2)
            #print(self.W1[0])

            # Step 2f: test network on testing data
            score = self.score()
            print(f"Epoch {i} complete")
            print(f"Score: {score}")

        # Step 3: compute final score of network over testing data
        final_score = self.score()
        print(f"Network performance: {final_score}")

    def forwardprop(self, input, weight, bias):
        return input @ weight + bias
    
    def score(self):

        Z1 = (self.X_test @ self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self.sigmoid(Z2)

        predictions = np.argmax(A2, axis=1)
        labels = np.argmax(self.y_test, axis=1)
        
        return np.mean(predictions == labels)

        '''correct = 0
        total = 0
        for x in self.X_test:
            h1 = self.forwardprop(x, self.W1, self.b1)
            h1 = self.relu(h1)
            #h2 = self.forwardprop(h1, self.W2, self.b2)
            y = self.forwardprop(h1, self.W3, self.b3)
            y = self.sigmoid(y)
            y = np.argmax(y) # what the model thinks it is
            yhat = np.where(self.y_test[total] == 1)[0] # what it actually is
            if y == yhat:
                correct = correct + 1
            total = total + 1
        return correct / total'''
    
    def cost(self, observed, expected):
        return np.sum((observed - expected) ** 2) / 2 # yhat is observed, y is expected
    
    def sigmoid(self, z):
        z = np.where(z > -709, z, -709) # prevents overflow encountered in exp lol
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.where(z > 0, z, 0)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)