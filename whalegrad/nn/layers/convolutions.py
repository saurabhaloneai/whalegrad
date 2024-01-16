from whalegrad.engine.functions import Action
import numpy as np

#convo2d, convo1d, pooling, max pooling 

class Convo1d:
    
    def __init__(self,x_features,y_features):
        self.x_features, self.y_features = x_features, y_features
        np.dot(x_features ,y_features )
        pass
    def forward(self):
        