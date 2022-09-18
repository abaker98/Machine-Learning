import os
import random
import traceback
from pdb import set_trace

import numpy as np

from itcs4156.util.timer import Timer
from itcs4156.util.data import feature_label_split, Standardization
from itcs4156.util.metrics import mse
from itcs4156.datasets.HousingDataset import HousingDataset

from itcs4156.assignments.regression.OrdinaryLeastSquares import OrdinaryLeastSquares
from itcs4156.assignments.regression.LeastMeanSquares import LeastMeanSquares
from itcs4156.assignments.regression.PolynomialRegression import PolynomialRegression
from itcs4156.assignments.regression.PolynomialRegressionRegularized import PolynomialRegressionRegularized
from itcs4156.assignments.regression.train import HyperParameters
from itcs4156.assignments.regression.train import get_features_for_lsm, get_features_for_poly_reg

def split_data(df_trn, df_vld, feature_names, label_name, return_df=False):
    X_trn, y_trn = feature_label_split(df_trn, 
                                       feature_names=feature_names, 
                                       label_name=label_name, 
                                       return_df=return_df)
    X_vld, y_vld = feature_label_split(df_vld, 
                                       feature_names=feature_names, 
                                       label_name=label_name, 
                                       return_df=return_df)
    
    return X_trn, y_trn, X_vld, y_vld

def standardize_data(X_trn, X_vld):
    standardize = Standardization()
    X_trn_clean = standardize.fit_transform(X_trn)
    X_vld_clean = standardize.transform(X_vld)
    
    return X_trn_clean, X_vld_clean

def get_cleaned_data(df_trn, df_vld, feature_names, label_name, return_df=False):
    X_trn, y_trn, X_vld, y_vld = split_data(df_trn, df_vld, feature_names, label_name)
    X_trn, X_vld = standardize_data(X_trn, X_vld)

    return X_trn, y_trn, X_vld, y_vld

class RunModel():
    t1 = '\t'
    t2 = '\t\t'
    t3 = '\t\t\t'
    def __init__(self, model, model_params):
        self.model_name = model.__name__
        self.model_params = model_params
        self.model = self.build_model(model, model_params)

    def build_model(self, model, model_params):
        print("="*50)
        print(f"Building model {self.model_name}")
        
        try:
            model = model(**model_params)
        except Exception as e:
            err = f"Exception caught while building model for {self.model_name}:"
            catch_and_throw(e, err)
        return model
    
    def train(self, *args, **kwargs):
        print(f"{self.t1}Training {self.model_name}...")
        print(f"{self.t2}Using hyperparameters: ")
        [print(f"{self.t3}{n} = {v}")for n, v in self.model_params.items()]
        try:
            return self._train(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while training model for {self.model_name}:"
            catch_and_throw(e, err)
            
    def _train(self, X, y, metrics=None, pass_y=False):
        if pass_y:
            self.model.train(X, y)
        else:
             self.model.train(X)
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix='Train')
        return scores
    
    def evaluate(self, *args, **kwargs):
        print(f"{self.t1}Evaluating {self.model_name}...")
        try:
            return self._evaluate(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while evaluating model for {self.model_name}:"
            catch_and_throw(e, err)
        

    def _evaluate(self, X, y, metrics, prefix=''):
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix)      
        return scores
    
    def predict(self, X):
        try:
            preds = self.model.predict(X)
        except Exception as e:
            err = f"Exception caught while making predictions for model {self.model_name}:"
            catch_and_throw(e, err)
            
        return preds
    
    def get_metrics(self, y, y_hat, metrics, prefix=''):
        scores = {}
        for name, metric in metrics.items():
            score = metric(y, y_hat)
            scores[name] = score
            print(f"{self.t2}{prefix} {name}: {score}")
        return scores

def run_eval(eval_stage='validation'):
    run_timer = Timer()
    run_timer.start()
    
    set_seeds(seed=25)
    
    avg_trn_mse = 0.0
    avg_vld_mse = 0.0
    successful_tests = 0
    
    models = [OrdinaryLeastSquares, LeastMeanSquares, PolynomialRegression, PolynomialRegressionRegularized]
    model_features = ["RM", get_features_for_lsm(), "LSTAT", get_features_for_poly_reg()]
    model_threshold_mse = [60, 90, 30, 30]
    metrics = {"MSE": mse}
    
    dataset = HousingDataset()
    df_trn, df_vld = dataset.load()

    for feature_names, model, threshold in zip(model_features, models, model_threshold_mse):
        try: 
            model_name = model.__name__
            model_params = HyperParameters.get_params(model_name)
            run_model = RunModel(model, model_params)
            
            if "MEDV" in feature_names:
                print("\nThe target feature 'MEDV' can not be used as an input feature!")
                print("Removing MEDV from your feature list and proceeding...\n")
                feature_names = [ f for f in feature_names if f != "MEDV"]
            

            X_trn, y_trn, X_vld, y_vld = get_cleaned_data(df_trn, df_vld, feature_names, "MEDV")

            trn_scores = run_model.train(X_trn, y_trn, metrics, pass_y=True)
            vld_scores = run_model.evaluate(X_vld, y_vld, metrics, prefix=eval_stage.capitalize())
        
            status = vld_scores['MSE'] < threshold
            print(f"\tChecking if {eval_stage.capitalize()} MSE {vld_scores['MSE']} < {threshold}: {status}")
            print("\tTest {}\n".format("PASSED! (っ＾▿＾)💨" if status else "FAILED! (ノಠ益ಠ)ノ彡┻━┻"))

            if status:
                avg_trn_mse += trn_scores['MSE']
                avg_vld_mse += vld_scores['MSE']
                successful_tests += 1
                
        except Exception as e:
            track = traceback.format_exc()
            print("The following exception occurred while executing this test case:\n", track)
    
    if successful_tests > 0:
            avg_trn_mse = avg_trn_mse / successful_tests
            avg_vld_mse = avg_vld_mse / successful_tests
    else:
        avg_trn_mse = 0
        avg_vld_mse = 0

    score = successful_tests * 20
    print("Tests passed: {}/4, Score: {}/80\n".format(successful_tests, score))
    print(f"MSE averages for {successful_tests} successful tests")
    print(f"\tAverage Training MSE: {avg_trn_mse}")
    print(f"\tAverage {eval_stage.capitalize()} MSE: {avg_vld_mse}")
    
    run_timer.stop()
    
    return score, avg_trn_mse, avg_vld_mse

def catch_and_throw(e, err):
    trace = traceback.format_exc()
    print(err + f"\n{trace}")
    raise e

def set_seeds(seed):
    np.random.seed(25)
    random.seed(25)

if __name__ == "__main__":
    run_eval()
  




    


