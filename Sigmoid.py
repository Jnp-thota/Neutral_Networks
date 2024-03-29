from Layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        self.output = None
    def forward(self,inputs):
        self.inputs = inputs
        inputs = inputs.astype(float)
        inputs = np.array(inputs)
        # print("Inputs type : ",inputs)
        self.outputs = 1/(1+np.exp(-inputs))
        # print("Inputs type : ",type(self.outputs))
        return self.outputs
    def backward(self,grad):
        sigmoid_grad = self.outputs * (1 - self.outputs)
        
        if type(grad) != np.ndarray:
            grad_concatenated = np.concatenate(grad, axis=None)
            grad = np.array(grad_concatenated)
        return sigmoid_grad * grad
        