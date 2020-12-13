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


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y. grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)
B = b.creator
a = B.input
a.grad = B.backward(b.grad)
A = a.creator
x = A.input
x.grad = A.backward(a.grad)

print(x.grad)

# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a
# assert y.creator.input.creator.input.creator == A
# assert y.creator.input.creator.input.creator.input == x

# def f(x):
#     A = Square()
#     B = Exp()
#     C = Square()

#     return C(B(A(x)))

# x = Variable(np.array(0.5))
# dy = numerical_diff(f, x)
# print(dy)

# f = Square()
# x = Variable(np.array(2))

# dy = numerical_diff(f, x)
# print(dy)



# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)

# print(x.grad)

# print(y.data)


