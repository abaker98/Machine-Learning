import numpy as np

from itcs4156.models.LinearModel import LinearModel

class OrdinaryLeastSquares(LinearModel): 
    """ 
        Performs regression using ordinary least squares
        
        attributes
        ===========
        w    nd.array  (column vector/matrix)
             weights
    """
    def __init__(self, lamb=0):
        super().__init__()
        self.lamb = lamb
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform OLS in order to learn the 
                weights `self.w`.
        """
        I = np.eye(X.shape[1])
        I[0,0] = 0
        self.w = np.linalg.pinv(self.add_ones(X).T @ self.add_ones(X) + self.lamb * I) @ self.add_ones(X).T @ y
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Used to make a prediction using the learned weights.
        
            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`.
        """
        return self.add_ones(X) @ self.w
