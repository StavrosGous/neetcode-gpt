import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.array(x)
        layers = len(weights)
        print(layers)
        z = 0
        prev = x
        for i in range(layers):
            print(i)
            z = np.dot(weights[i].T, prev) + biases[i]
            if i == layers-1: break
            z = np.maximum(z, 0)
            prev = z
        return np.round(z, 5)