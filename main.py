import numpy as np
from Variable import Variable
from Function import Function

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)

# x = Variable(1.0)
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)

print(y.data)



