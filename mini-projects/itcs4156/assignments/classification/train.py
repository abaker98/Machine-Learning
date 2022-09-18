import numpy as np
from sklearn.pipeline import Pipeline

from itcs4156.util.data import binarize_classes, dataframe_to_array
from itcs4156.util.data import AddBias, Standardization, ImageNormalization, OneHotEncoding

class HyperParametersAndTransforms():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParametersAndTransforms, name)
        params = {}
        for key, value in model.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if not callable(value) and not isinstance(value, staticmethod):
                    params[key] = value
        return params
    
    class Perceptron():
        alpha = 0.5 
        epochs = 5

        def __init__(self):
            self.target_pipe = None # TODO (OPTIONAL) Add Pipeline() definitions below
            self.feature_pipe = Pipeline([('scalar', Standardization()), ('bias', AddBias())])
        
        @staticmethod
        def data_prep(dataset, return_array=False):
            X_trn_df, y_trn_df, X_vld_df, y_vld_df = dataset.load()

            X_trn_df, y_trn_df = binarize_classes(
                X_trn_df, 
                y_trn_df, 
                pos_class=[1],
                neg_class=[0], 
            )
            
            X_vld_df, y_vld_df = binarize_classes(
                X_vld_df, 
                y_vld_df, 
                pos_class=[1], 
                neg_class=[0], 
            )
            
            perceptron_transform = HyperParametersAndTransforms.Perceptron()
            X_trn_df, y_trn_df = perceptron_transform.fit_transform(X=X_trn_df, y=y_trn_df)
            X_vld_df, y_vld_df = perceptron_transform.transform(X=X_vld_df, y=y_vld_df)
            
            if return_array:
                print("Returning data as NumPy array...")
                return dataframe_to_array([X_trn_df, y_trn_df, X_vld_df, y_vld_df])
            
            return X_trn_df, y_trn_df, X_vld_df, y_vld_df
        
        
        def fit(self, X, y=None):
            if self.target_pipe  is not None:
                self.target_pipe.fit(y)
                
            if self.feature_pipe is not None:
                self.feature_pipe.fit(X)

        def transform(self, X, y=None):
            if self.target_pipe is not None:
                y = self.target_pipe.transform(y)
                
            if self.feature_pipe is not None:
                X = self.feature_pipe.transform(X)
            return X, y
        
        def fit_transform(self, X, y):
            self.fit(X, y)
            X, y = self.transform(X, y)
            return X, y
        
    class NaiveBayes():
        smoothing = 10e-2 # (OPTIONAL) TODO Set Gaussian Naive Bayes smoothing
        
        def __init__(self):
            self.target_pipe = None # TODO (OPTIONAL) Add Pipeline() definitions below
            self.feature_pipe = Pipeline([('scalar', Standardization())])
            
        @staticmethod
        def data_prep(dataset, return_array=False):
            X_trn_df, y_trn_df, X_vld_df, y_vld_df = dataset.load()

            transforms = HyperParametersAndTransforms.NaiveBayes()
            X_trn_df, y_trn_df = transforms.fit_transform(X=X_trn_df, y=y_trn_df)
            X_vld_df, y_vld_df = transforms.transform(X=X_vld_df, y=y_vld_df)
            
            if return_array:
                print("Returning data as NumPy array...")
                return dataframe_to_array([X_trn_df, y_trn_df, X_vld_df, y_vld_df])
            
            return X_trn_df, y_trn_df, X_vld_df, y_vld_df
        
        def fit(self, X, y=None):
            if self.target_pipe  is not None:
                self.target_pipe.fit(y)
                
            if self.feature_pipe is not None:
                self.feature_pipe.fit(X)

        def transform(self, X, y=None):
            if self.target_pipe  is not None:
                y = self.target_pipe.transform(y)
                
            if self.feature_pipe is not None:
                X = self.feature_pipe.transform(X)
            return X, y
        
        def fit_transform(self, X, y):
            self.fit(X, y)
            X, y = self.transform(X, y)
            return X, y
        
    class LogisticRegression():
        alpha = 0.1
        epochs = 100
        batch_size = None # TODO (OPTIONAL) Set LogisticRegression's mini-batch size if using mini-batch gradient descent
        
        def __init__(self):
            self.target_pipe = Pipeline([('labels', OneHotEncoding())])
            self.feature_pipe = Pipeline([('scalar', Standardization()),('bias' , AddBias())])
            
        @staticmethod
        def data_prep(dataset, return_array=False):
            X_trn_df, y_trn_df, X_vld_df, y_vld_df = dataset.load()

            transforms = HyperParametersAndTransforms.LogisticRegression()
            X_trn_df, y_trn_df = transforms.fit_transform(X=X_trn_df, y=y_trn_df)
            X_vld_df, y_vld_df = transforms.transform(X=X_vld_df, y=y_vld_df)

            if return_array:
                print("Returning data as NumPy array...")
                return dataframe_to_array([X_trn_df, y_trn_df, X_vld_df, y_vld_df])
            
            return X_trn_df, y_trn_df, X_vld_df, y_vld_df

        def fit(self, X, y=None):
            if self.target_pipe  is not None:
                self.target_pipe.fit(y)
                
            if self.feature_pipe is not None:
                self.feature_pipe.fit(X)

        def transform(self, X, y=None):
            if self.target_pipe is not None:
                y = self.target_pipe.transform(y)
                
            if self.feature_pipe is not None:
                X = self.feature_pipe.transform(X)
            return X, y
        
        def fit_transform(self, X, y):
            self.fit(X, y)
            X, y = self.transform(X, y)
            return X, y
        