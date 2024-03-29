from Layer import Layer
import numpy as np

class ReLu(Layer):
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        return np.maximum(0, x)
    
    def backward(self, grad):
        grad_input = grad if isinstance(grad, np.ndarray) else grad[0]
        grad_input[self.mask] = 0
        return grad_input