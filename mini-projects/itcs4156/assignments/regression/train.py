import importlib

class HyperParameters():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParameters, name)
        return {key:value for key, value in model.__dict__.items() 
            if not key.startswith('__') and not callable(key)}
    
    class OrdinaryLeastSquares():
        pass # No hyperparamters to set
        
    class LeastMeanSquares():
        alpha = None # TODO Set your learning rate
        
    class PolynomialRegression():
        degree = None # TODO Set your polynomial degree
        
    class PolynomialRegressionRegularized():
        degree = None # TODO Set your polynomial degree
        lamb = None # TODO Set your regularization value for lambda

def get_features_for_lsm():
    return [] # TODO fill this list with features you want to use for training LSM
    
def get_features_for_poly_reg():
    return [] # TODO fill this list with features you want to use for training PolynomialRegressionRegularized
