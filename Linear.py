
from Layer import Layer
import numpy as np


class Linear(Layer):
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights  = np.random.randn(output_size,input_size) *0.01
        self.bias = np.random.randn(output_size)
        
    def forward(self,inputs):
        print(type(inputs))
        self.inputs = inputs
        data = np.dot(inputs,self.weights.T)
        print(type(data))
        return data + self.bias
    
    def backward(self, grad):
        input_grad = np.dot(grad, self.weights.T)
        weights_grad = np.dot(self.inputs.T, grad)
        self.weights -= 0.01 * weights_grad
        self.bias -= 0.01 * np.sum(grad, axis=0, keepdims=True)
        return input_grad
        
        # input_gradient = np.dot(grad, self.weights)
        # weights_gradient = np.dot(grad.T,self.inputs)
        # bias_gradient = np.sum(grad, axis=0)

        # self.weights -= weights_gradient*0.01
        # self.bias -= bias_gradient*0.01

        # return input_gradient

        # # Gradient with respect to inputs
        # grad_inputs = np.dot(grad, self.weights)
        
        # # Gradient with respect to weights
        # grad_weights = np.dot(grad.T, self.inputs)
        
        # # Gradient with respect to bias
        # grad_bias = np.sum(grad, axis=0)

        # return grad_inputs
                