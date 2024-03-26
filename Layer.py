class Layer():
    def __init__(self):
        self.inputs = None
        self.outputs = None
        
    def forward(self,inputs):
        raise NotImplementedError
    
    def backward(serlf,grad):
        raise NotImplementedError