
import random
import traceback
from pdb import set_trace

from sklearn.metrics import accuracy_score
import numpy as np

from itcs4156.datasets.MNISTDataset import MNISTDataset
from itcs4156.util.timer import Timer

from itcs4156.assignments.classification.LogisticRegression import LogisticRegression
from itcs4156.assignments.classification.Perceptron import Perceptron
from itcs4156.assignments.classification.NaiveBayes import NaiveBayes
from itcs4156.assignments.classification.train import HyperParametersAndTransforms as hpt

def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    set_seeds(seed=25)
    
    models = [Perceptron, NaiveBayes, LogisticRegression]
    rubrics = [rubric_perceptron, rubric_naive_bayes, rubric_logistic_regression]
    metrics = {"Accuracy": accuracy}

    dataset = MNISTDataset()
    total_points = 0
    accs = []
    for model, rubric in zip(models, rubrics):
        try:
            timer =  Timer()
            model_name = model.__name__
            model_params = hpt.get_params(model_name)
            model_hpt = getattr(hpt, model_name)
            
            timer.start()
            run_model = RunModel(model, model_params)
            X_trn, y_trn, X_vld, y_vld = model_hpt.data_prep(dataset, return_array=True)
            
            trn_scores = run_model.train(X_trn, y_trn, metrics, pass_y=True)
            vld_scores = run_model.evaluate(X_vld, y_vld, metrics, prefix=eval_stage.capitalize())

        except Exception as e:
            vld_scores = {'Accuracy': 0}
            track = traceback.format_exc()
            print("The following exception occurred while executing this test case:\n", track)
        timer.stop()
        
        print("")
        points = rubric(vld_scores['Accuracy'])
        print(f"Points Earned: {points}")
        total_points += points
        accs.append(vld_scores['Accuracy'])
        
    print("="*50)
    main_timer.stop()
    total_points = round(total_points)
    print(f"Totals Points Earned: {round(total_points)}/80")
    
    return total_points, main_timer.last_elapsed_time, accs

def set_seeds(seed):
    np.random.seed(25)
    random.seed(25)

def accuracy(y, y_hat):
    # Convert y from one-hot encoding back to normal
    if len(y.shape) > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=1).reshape(-1,1)
    # Reshape labels and preds to be 2D arrays
    elif len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(y_hat.shape) == 1:
        y_hat = y_hat.reshape(-1, 1)
    
    return accuracy_score(y, y_hat)
  
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
            display_score = round(score, 3)
            scores[name] = score
            print(f"{self.t2}{prefix} {name}: {display_score}")
        return scores
    
def catch_and_throw(e, err):
    trace = traceback.format_exc()
    print(err + f"\n{trace}")
    raise e

def rubric_perceptron(acc, max_score=25):
    score_percent = 0
    if acc >= 0.8:
        score_percent = 100
    elif acc >= 0.75:
        score_percent = 90
    elif acc >= 0.70:
        score_percent = 80
    elif acc >= 0.65:
        score_percent = 70
    elif acc >= 0.60:
        score_percent = 60
    elif acc >= 0.55:
        score_percent = 50
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def rubric_naive_bayes(acc, max_score=25):
    score_percent = 0
    if acc >= 0.70:
        score_percent = 100
    elif acc >= 0.60:
        score_percent = 90
    elif acc >= 0.50:
        score_percent = 80
    elif acc >= 0.40:
        score_percent = 70
    elif acc >= 0.30:
        score_percent = 60
    elif acc >= 0.20:
        score_percent = 50
    elif acc >= 0.10:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score
   
def rubric_logistic_regression(acc, max_score=30):
    score_percent = 0
    if acc >= 0.80:
        score_percent = 100
    elif acc >= 0.70:
        score_percent = 90
    elif acc >= 0.60:
        score_percent = 80
    elif acc >= 0.50:
        score_percent = 70
    elif acc >= 0.40:
        score_percent = 60
    elif acc >= 0.30:
        score_percent = 55
    elif acc >= 0.20:
        score_percent = 50
    elif acc >= 0.15:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

if __name__ == "__main__":
    run_eval()

