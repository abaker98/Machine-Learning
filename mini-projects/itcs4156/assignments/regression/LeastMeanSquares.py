import numpy as np

from itcs4156.models.LinearModel import LinearModel

class LeastMeanSquares(LinearModel):
    """
        Performs regression using least mean squares (gradient descent)
    
        attributes:
            w (np.ndarray): weight matrix
            
            alpha (float): learning rate or step size
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.w = None
        self.alpha = alpha
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform LMS in order to learn the 
                weights `self.w`.
        """
        pass # TODO replace this line with your code
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Used to make a prediction using the learned weights.
        
            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`.
        """
        pass # TODO replace this line with your code
