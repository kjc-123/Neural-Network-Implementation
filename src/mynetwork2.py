import numpy as np
import loader

# TODO: MORE HIDDEN LAYERS!!!!!!

# TODO: should sigmoid be in forward prop? should initial values be N(0,1) or U(0,1)?

class MyNetwork():

    # used ChatGPT for explanation of steps
    # Step 1: initialize weights and biases
    # TODO: generalize size and number of hidden layers
    # TODO: replace sigmoid with ReLu
    # TODO: add second hidden layer W2
    W1 = np.random.randn(784, 16) # randn is standard normal distribution Z ~ N(0, 1)
    W2 = np.random.randn(16, 16)
    W3 = np.random.randn(16, 10)
    b1 = np.random.randn(1, 16)
    b2 = np.random.randn(1, 16)
    b3 = np.random.randn(1, 10)

    ld = loader.loader()

    X_train, y_train = ld.load_data("data/mnist_train.csv")
    X_test, y_test = ld.load_data("data/mnist_test.csv")

    def __init__(self):
        print("network initialized")
    
    def sgd(self):

        eta = 3.0 # learning rate
        W1 = self.W1
        W2 = self.W2
        W3 = self.W3
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        m = 6000

        for i in range(0, 10): # 10 epochs

            # Step 2a: sample minibatch of size m from training data
            i0 = (m * i)
            il = (m * (i+1))
            X_batch = self.X_train[i0:il,:]
            y_batch = self.y_train[i0:il]

            # Step 2b: forward pass
            A1 = self.forwardprop(X_batch, W1, np.ones([m,1]) @ b1)
            A2 = self.forwardprop(A1, W2, np.ones([m,1]) @ b2)
            A3 = self.forwardprop(A2, W3, np.ones([m,1]) @ b3)

            # Step 2c: backward pass
            del3 = np.multiply(A3 - y_batch, np.multiply(A3, np.ones(A3.shape)-A3))
            del2 = np.multiply(del3 @ np.transpose(W3), np.multiply(A2, np.ones(A2.shape)-A2))
            del1 = np.multiply(del2 @ np.transpose(W2), np.multiply(A1, np.ones(A1.shape)-A1))

            # Step 2d: calculate average gradients over batch
            delW3 = (np.transpose(A1) @ del3) / m
            delb3 = np.ones([1, m]) @ del3
            delW2 = (np.transpose(A1) @ del2) / m
            delb2 = np.ones([1, m]) @ del2
            delW1 = (np.transpose(X_batch) @ del1) / m
            delb1 = np.ones([1, m]) @ del1

            # Step 2e: update parameters
            #print(del1[0])
            self.W1 = self.W1 - (eta * delW1)
            self.b1 = self.b1 - (eta * delb1)
            #self.W2 = self.W2 - (eta * delW2)
            #self.b2 = self.b2 - (eta * delb2)
            self.W3 = self.W3 - (eta * delW3)
            self.b3 = self.b3 - (eta * delb3)

            # Step 2f: test network on testing data
            score = self.score()
            print(f"Epoch {i} complete")
            print(f"Score: {score}")

        # Step 3: compute final score of network over testing data
        final_score = self.score()
        print(f"Network performance: {final_score}")

    def forwardprop(self, input, weight, bias):
        return self.sigmoid(input @ weight + bias)
    
    def backprop():
        # TODO
        print("TODO")
    
    # TODO: generalize number of hidden layers
    def score(self):
        correct = 0
        total = 0
        for x in self.X_test:
            h1 = self.forwardprop(x, self.W1, self.b1)
            h2 = self.forwardprop(h1, self.W2, self.b2)
            y = self.forwardprop(h2, self.W3, self.b3)
            y = np.argmax(y) # what the model thinks it is
            yhat = np.where(self.y_test[total] == 1) # what it actually is
            if y == yhat:
                correct = correct + 1
            total = total + 1
        return correct / total
    
    def average_cost():
        # TODO 
        print("TODO")
    
    def cost(self, observed, expected):
        return np.sum((observed - expected) ** 2) / 2 # yhat is observed, y is expected
    
    def sigmoid(self, z):
        z = np.where(z > -709, z, -709) # prevents overflow encountered in exp lol
        return 1 / (1 + np.exp(-z))