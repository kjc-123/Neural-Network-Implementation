# loads mnist csv file and returns numpy matrix of data
# will convert first value in each vector (a digit between 0-9) into corresponding array
# ex: 2.0 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# this makes computing cost later simpler

import numpy as np

class loader():
    
    def __init__(self):
        return None

    def load_data(self, file_location):
        data = np.loadtxt(file_location, delimiter=',')
        input = data[:,1:785]
        # vectorize the digits 0-9
        output = [int(i) for i in data[:,0]]
        output = np.array([self.vectorized_result(y) for y in output])
        return input, output
    
    # turns integer x into vector with 1 at place x and 0 everywhere else
    def vectorized_result(self, i):
        result = np.zeros(10)
        result[i] = 1.0
        return result