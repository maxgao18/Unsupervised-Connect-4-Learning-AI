import numpy as np

# leaky relu function
class LeakyRELU:
    # function
    @staticmethod
    def func (z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return z
            return 0.1*z
        elif z.dtype == np.int:
            z = np.asfarray(z, dtype='float')

        for i, zi in enumerate(z):
            z[i] = LeakyRELU.func(zi)
        return z

    # Derivative for leaky relu
    @staticmethod
    def func_deriv(z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return 1.0
            return 0.1
        for i, zi in enumerate(z):
            z[i] = LeakyRELU.func_deriv(zi)
        return z

class RELU:
    # function
    @staticmethod
    def func (z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return z
            return 0.0
        elif z.dtype == np.int:
            z = np.asfarray(z, dtype='float')

        for i, zi in enumerate(z):
            z[i] = RELU.func(zi)
        return z

    # Derivative for leaky relu
    @staticmethod
    def func_deriv(z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return 1.0
            return 0.0
        for i, zi in enumerate(z):
            z[i] = RELU.func_deriv(zi)
        return z

class Sigmoid:
    # function
    @staticmethod
    def func (z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 15:
                return 0.999999999
            elif z < -15:
                return 0.000000001
            return np.exp(z)
        elif z.dtype == np.int:
            z = np.asfarray(z, dtype='float')

        for i, zi in enumerate(z):
            z[i] = Sigmoid.func(zi)
        return z

    # func derivative
    @staticmethod
    def func_deriv (z):
        if isinstance(z, float) or isinstance(z, int):
            z = Sigmoid.func(z)
            return z*(1-z)
        for i, zi in enumerate(z):
            z[i] = Sigmoid.func_deriv(zi)
        return z

# Softmax function
class Softmax:
    # used to raise powers to e
    @staticmethod
    def get_exp(z):
        if isinstance(z, float) or isinstance(z, int):
            return np.exp(z)
        elif z.dtype == np.int:
            z = np.asfarray(z, dtype='float')

        for i, zi in enumerate(z):
            z[i] = Softmax.get_exp(zi)
        return z

    # softmax func
    @staticmethod
    def func(z):
        z = Softmax.get_exp(z)
        return z / np.sum(z)

    #derivative of softmax (z*(1-z)) for unsquashed activations z
    @staticmethod
    def func_deriv(z):
        z = Softmax.func(z)
        return z*(1-z)

class CustomActivation:
    @staticmethod
    def func(z):
        z[:7] = Softmax.func(z[:7])
        z[7:] = Sigmoid.func(z[7:])
        return z

    @staticmethod
    def func_deriv(z):
        if z.dtype == np.int:
            z = np.asfarray(z, dtype='float')
        z[:7] = Softmax.func_deriv(z[:7])
        z[7:] = Sigmoid.func_deriv(z[7:])
        return z
