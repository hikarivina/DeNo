import numpy as np
from Variable import Variable


class Function:
    def __call__(self, *inputs):

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_ndarray(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.input = input
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

def as_ndarray(x):
    if np.isscalar(x):
        return np.array(x)
    return x