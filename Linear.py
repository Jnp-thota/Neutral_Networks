
from Layer import Layer
import numpy as np


class Linear(Layer):
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights  = np.random.randn(output_size,input_size)
        self.bias = np.zeros(output_size)
        
    def forward(self,inputs):
        # print(type(inputs))
        self.inputs = inputs
        data = np.dot(inputs,self.weights.T)
        # print(type(data))
        return data + self.bias
    
    def backward(self, grad):
        grad_input = np.dot(grad, self.weights)
        self.grad_weights = np.dot(grad.T, self.inputs).astype(np.float64)
        self.grad_bias = np.sum(grad, axis=0)
        return grad_input, self.grad_weights, self.grad_bias
                