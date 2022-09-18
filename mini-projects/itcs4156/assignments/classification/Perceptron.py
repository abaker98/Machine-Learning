import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class Perceptron(ClassificationModel):
    """
        Performs Gaussian Naive Bayes
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs (int): Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            batch_size (int): Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.w = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """ Train model to learn optimal weights for performing binary classification.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
             TODO:
                Finish this method by using Rosenblatt's Perceptron algorithm to learn
                the best weights to classify the binary data. There is no need to
                implement th pocket algorithm unless you choose to do so. Also, update 
                and store the learned weights into `self.w`.
        """
        m_samples = X.shape[0]
        n_features = X.shape[1]
         
        rng = np.random.RandomState(42)
        self.w = rng.rand(n_features)
        
        for e in range(self.epochs):
            misclassified = 0
            
            for i in range(m_samples):
                
                z = self.w @ X[i]
                 
                y_hat = np.sign(z)
                
                if y_hat != y[i]:
                    self.w = self.w + self.alpha * y[i] * X[i]
                    misclassified += 1
            
            if misclassified == 0:
                print(f"Converged at epoch: {e} - No samples misclassified")
                break
        print(f"Epochs trained: {e+1}")
   
    def predict(self, X: np.ndarray):
        """ Used to make a prediction using the learned weights.
        
            Args:
                X: Data 

            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # TODO Add code below
        y_hat = np.sign(X @ self.w)
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)

