import json
import math
import numpy as np
import scipy as sp

train_file = open('mnist_handwritten_train.json', 'r')
test_file = open('mnist_handwritten_test.json', 'r')

train_data = json.load(train_file)

X = sp.array([ex['image'] for ex in train_data])
X = X/255 # normalize

y = sp.array([ex['label'] for ex in train_data])

N = len(X)
L = 10 # number of labels

'''
y is the label for each example, so y would look something like:
    [5, 9, 4, ...]

which means that ex0 is an image of the number 5, ex1 is 9, ex2 is 4, etc.,
but, in order to calculate the error and do backpropagation I need to
restructure this data, because my expected output should look like:

          0 1 2 3 4 5 6 7 8 9
    ex0: [0,0,0,0,0,1,0,0,0,0]
    ex1: [0,0,0,0,0,0,0,0,0,1]
    ex2: [0,0,0,0,1,0,0,0,0,0]
    ...

This means, in a *perfect world* my classifier will output 0 for all of the
digits it isnt, and it will output 1 for the correct digit.

`expected` accounts for this; it only has 1 at the correct index and 0
everywhere else
'''
expected = sp.zeros([N,L])
expected[sp.arange(N), y] = 1 # is only 1 for the correct digit

# weights for hidden neuron inputs, w0 is bias
# 784 inputs + 1 bias, 30 hidden neurons
W1 = sp.random.rand(785, 30)

# weights for output neuron inputs, w0 is bias
# 30 inputs + 1 bias, 10 output labels
W2 = sp.random.rand(31, 10)

# activation function
sigm = lambda x: 1/(1+sp.exp(-x))

# HIDDEN NEURON
input_1 = sp.insert(X, 0, 1, axis=1)            # put 1 in col0 for bias
weighted_1 = input_1.dot(W1)                    # apply weights
max_1 = np.max(weighted_1, axis=1)              # max per row
norm_1 = weighted_1/max_1[:, sp.newaxis]        # normalize by dividing by max
output_1 = sigm(norm_1)                         # apply activation function

# OUTPUT NEURON
input_2 = sp.insert(output_1, 0, 1, axis=1)     # put 1 in col0 for bias
weighted_2 = input_2.dot(W2)                    # apply weights
max_2 = np.max(weighted_2, axis=1)              # max per row
norm_2 = weighted_2/max_2[:, sp.newaxis]        # normalize by dividing by max
output_2 = sigm(norm_2)                         # apply activation function

# output_2 is now the predicted output

err = output_2 - expected
