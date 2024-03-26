from Layer import Layer
import numpy as np

class BinaryCrossEntropyLoss(Layer):
    def forward(self,inputs,targets):
        self.inputs = inputs
        
        targets = targets.values.flatten()
        self.targets = targets
        epsilon = 1e-15
        self.logits = np.clip(inputs,epsilon,1-epsilon)
        loss = -(targets * np.log(inputs) + (1 - targets) * np.log(1 - inputs))
        return np.mean(loss) 
    
    def backward(self,grad):
        epsilon = 1e-10
        clipped_y_pred = np.clip(self.inputs, epsilon, 1 - epsilon)
        input_grad = (clipped_y_pred - self.targets) / (clipped_y_pred * (1 - clipped_y_pred))
        return input_grad