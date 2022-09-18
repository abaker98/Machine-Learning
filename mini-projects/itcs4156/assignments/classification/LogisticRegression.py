from typing import List, Tuple, Union 

import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class LogisticRegression(ClassificationModel):
    """
        Performs Logistic Regression using the softmax function.
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs: Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            batch_size: Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int, batch_size: int):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
    
    def init_weights(self, X, y):
        rng = np.random.RandomState(42)
        self.w = rng.rand(X.shape[1], y.shape[1])

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """ Computes probabilities for multi-class classification given continuous inputs z.
        
            Args:
                z: Continuous outputs after dotting the data with the current weights 

            TODO:
                Finish this method by adding code to return the softmax. Don't forget
                to subtract the max from `z` to maintain  numerical stability!
        """
        z = z - np.max(z, axis=-1, keepdims=True)
        
        e_z = np.exp(z)
        
        denominator = np.sum(e_z, axis=-1, keepdims=True)
        
        softmax = e_z/denominator
        
        return softmax

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Train our model to learn optimal weights for classifying data.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
             TODO:
                Finish this method by using either batch or mini-batch gradient descent
                to learn the best weights to classify the data. You'll need to finish and 
                also call the `softmax()` method to complete this method. Also, update 
                and store the learned weights into `self.w`. 
        """
        self.init_weights(X, y)
        for e in range (self.epochs):
            z = X @ self.w
            probs = self.softmax(z)
            avg_gradient = (X.T @ (probs - y)) / len(y)  
            self.w -= self.alpha * avg_gradient
       
    def predict(self, X: np.ndarray):
        """ Used to make a prediction using the learned weights.
        
            Args:
                X: Data 

            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # TODO Add code below
        z = X @ self.w
        probs = self.softmax(z)
        y_hat = np.argmax(probs, axis=1)
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)
