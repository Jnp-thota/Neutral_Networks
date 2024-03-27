from Layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        self.output = None
    def forward(self,inputs):
        self.inputs = inputs
        inputs = inputs.astype(float)
        inputs = np.array(inputs)
        # if isinstance(inputs, (int, float)):
        #     inputs = np.array([[inputs]])
        # print("Inputs type : ",inputs)
        self.outputs = 1/(1+np.exp(-inputs))
        # print("Inputs type : ",type(self.outputs))
        return self.outputs
    def backward(self,grad):
        sigmoid_grad = self.outputs * (1 - self.outputs)
        # print("type of outputs",type(sigmoid_grad))
        # print("type of grad",type(grad))
        
        if type(grad) != np.ndarray:
            grad_concatenated = np.concatenate(grad, axis=None)
            grad = np.array(grad_concatenated)
            # print("the type: ", type(grad))
            # print("Shape of sigmoid_grad:", sigmoid_grad.shape)
            # print("Shape of grad:", grad.shape)
        # grad = np.array(grad)  # Convert grad to numpy array
        # print("Data type of grad:", grad.dtype)
        # print("grad:", grad)
        # print("Shape of grad:", grad.shape)
        # print("Shape of sigmoid_grad:", sigmoid_grad.shape)
        return sigmoid_grad * grad
        