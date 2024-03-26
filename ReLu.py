from Layer import Layer
import numpy as np

class ReLu(Layer):
    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs
    def backward(self,grad):
        relu_grad = np.where(self.inputs >= 0, 1, 0)
        return grad * relu_grad