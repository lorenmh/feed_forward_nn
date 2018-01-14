import json
import math
import numpy as np
import scipy as sp

train_file = open('mnist_handwritten_train.json', 'r')
test_file = open('mnist_handwritten_train.json', 'r')

train_data = json.load(train_file)

sigm = lambda x: 1/(1+sp.exp(-x))

# 60000x784 input matrix
X = sp.array([ex['image'] for ex in train_data])
X = X/255 # normalize

y = sp.array([ex['label'] for ex in train_data])

N = len(X)
L = 10 # number of labels

expected = sp.zeros([N,L])
expected[sp.arange(N), y] = 1

# weights for first layer 30x785
W1 = sp.random.rand(785, 30)

# weights for second layer 31x10
W2 = sp.random.rand(31, 10)

#y = np.argmax(some_output, axis=1)

input_1 = sp.insert(X, 0, 1, axis=1) # bias
apply_1 = input_1.dot(W1)
max_1 = np.max(apply_1, axis=1)
normalized_1 = apply_1/max_1[:, sp.newaxis]
output_1 = sigm(normalized_1)

input_2 = sp.insert(output_1, 0, 1, axis=1) # bias
apply_2 = input_2.dot(W2)
max_2 = np.max(apply_2, axis=1)
normalized_2 = apply_2/max_2[:, sp.newaxis]
output_2 = sigm(normalized_2)

err = output_2 - expected
