import numpy as np

class Accuracy:
    
    def __init__(self, y_preds, y_true):
        self.y_preds, self.y_true = y_preds, y_true
        
    def forward(self):
        
        return (np.mean(self.y_preds == self.y_true)) * 100     


# def acc(y_preds, y_true):
    
#     return (np.mean(y_preds == y_true)) * 100