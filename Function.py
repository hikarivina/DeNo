import numpy as np
from Variable import Variable


class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(as_ndarray(y))
        output.set_creator(self)
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

def as_ndarray(x):
    if np.isscalar(x):
        return np.array(x)
    return x