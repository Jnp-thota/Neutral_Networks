from Layer import Layer
import numpy as np

class Sequential(Layer):
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self,inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self,grad):
        for layer in reversed(self.layers):
            print("type of layer:",type(layer))
            print("type of grad",type(grad))
            grad = layer.backward(grad)
        return grad
    
    def save_weights(self, filename):
        weights = [layer.weights for layer in self.layers if hasattr(layer, 'weights')]
        biases = [layer.bias for layer in self.layers if hasattr(layer, 'bias')]
        
        # print("Shape of weights:",(len(weights),len(weights[0])))
        # print("Shape ofbiases:",(len(biases),len(biases[0])))
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            np.savez(f"layer_{i}_weights.npz", weights=weight)
            np.savez(f"layer_{i}_biases.npz", biases=bias)

    def load_weights(self, filename):
        data = np.load(filename)
        for layer, weights, biases in zip(self.layers, data['weights'], data['biases']):
            layer.weights = weights
            layer.bias = biases