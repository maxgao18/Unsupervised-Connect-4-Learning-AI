import numpy as np

# Quadratic cost function
class QuadraticCost:
    @staticmethod
    def cost (network_output, expected_output):
        return sum(0.5*(np.power(network_output-expected_output, 2)))

    @staticmethod
    def delta (network_output, z_activation_deriv, expected_output):
        return 0.5*(np.power(network_output-expected_output, 2)*z_activation_deriv)

# Optimized with softmax, but can be used with other functions
class NegativeLogLikelihood:
    @staticmethod
    def cost (network_output, expected_output):
        return sum(-1*(np.log(network_output)*expected_output))

    @staticmethod
    def delta (network_output, z_activation_deriv, expected_output):
        return -(expected_output/network_output)*z_activation_deriv

class CustomCost:
    @staticmethod
    def cost (network_output, expected_output):
        return NegativeLogLikelihood.cost(network_output[:7], expected_output[:7]) \
               + QuadraticCost.cost(network_output[7:], expected_output[7:])

    @staticmethod
    def delta (network_output, z_activation_deriv, expected_output):
        deltas = np.zeros(8)
        deltas[:7] = NegativeLogLikelihood.delta(network_output[:7], z_activation_deriv[:7], expected_output[:7])
        deltas[7:] = QuadraticCost.delta(network_output[7:], z_activation_deriv[7:], expected_output[7:])
        return deltas
