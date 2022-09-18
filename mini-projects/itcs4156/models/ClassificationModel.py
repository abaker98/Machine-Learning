from abc import abstractclassmethod

import numpy as np

from itcs4156.models.BaseModel import BaseModel

class ClassificationModel(BaseModel):
    """
        Abstract class for classification 
        
        Attributes
        ==========
    """

    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError(''.join(["Wrong matrix ", name]))

    ####################################################
    #### abstract funcitons ############################
    @abstractclassmethod
    def train(self, X, y):
        pass
    
    @abstractclassmethod
    def predict(self, y):
        pass 