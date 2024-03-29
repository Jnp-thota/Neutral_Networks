from Layer import Layer
import numpy as np

class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
        self.y_true = y_true
        loss = - (y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)
    
    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("True or predicted values are not set.")
        # Compute gradient of loss with respect to y_pred
        grad_loss = - (self.y_true / self.y_pred - (1 - self.y_true) / (1 - self.y_pred))
        return grad_loss