# Python Imports
import numpy as np
import random
import traceback
from pdb import set_trace

from sklearn.metrics import accuracy_score

# Internal Imports
from itcs4156.util.timer import Timer
from itcs4156.util.metrics import mse, accuracy
from itcs4156.datasets.DataPreparation import HousingDataPreparation
from itcs4156.datasets.DataPreparation import MNISTDataPreparation
from itcs4156.assignments.neural_networks.NeuralNetwork import NeuralNetwork
from itcs4156.assignments.neural_networks.NeuralNetworkRegressor import NeuralNetworkRegressor
from itcs4156.assignments.neural_networks.NeuralNetworkClassifier import NeuralNetworkClassifier
from itcs4156.assignments.neural_networks.train import HyperParametersAndTransforms as hpt

def housing_data_prep(**kwargs):
    housing_data_prep = HousingDataPreparation(**kwargs)
    return housing_data_prep.data_prep(return_array=True)

def mnist_data_prep(**kwargs):
    mnist_data_prep = MNISTDataPreparation(**kwargs)
    return mnist_data_prep.data_prep(return_array=True)

def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    task_models = [{'Regressor': NeuralNetworkRegressor}, {'Classifier': NeuralNetworkClassifier}]
    task_data_preps = [housing_data_prep, mnist_data_prep]
    task_metrics = [{"MSE": mse}, {"Accuracy": accuracy}]
    task_rubrics = [{'MSE': rubric_regression}, {"Accuracy": rubric_classification}]
    tasks = zip(task_models, task_data_preps, task_metrics, task_rubrics)
    
    total_points = 0
    task_vld_scores = {"MSE": np.float('inf'), "Accuracy": 0}
    for model, data_prep, metrics, rubric in tasks:
        task_timer =  Timer()
        task_timer.start()
        try:
            (task_name, model), = model.items()
            model_class_name = model.__name__
            (metric_name, metric_func), = metrics.items()
            (rubric_metric_name, rubric), = rubric.items()

            params = hpt.get_params(model_class_name)
            model_kwargs = params['model_kwargs']
            data_prep_kwargs = params['data_prep_kwargs']
            
            run_model = RunModel(model, model_kwargs)
            X_trn, y_trn, X_vld, y_vld = data_prep(**data_prep_kwargs)
            
            trn_scores = run_model.fit(X_trn, y_trn, metrics, pass_y=True)
            vld_scores = run_model.evaluate(X_vld, y_vld, metrics, prefix=eval_stage.capitalize())
            
            # Store metrics
            task_vld_scores[task_name] = vld_scores
            
        except Exception as e:
            track = traceback.format_exc()
            print("The following exception occurred while executing this test case:\n", track)
        task_timer.stop()
        
        print("")
        points = rubric(task_vld_scores[task_name][rubric_metric_name])
        print(f"Points Earned: {points}")
        total_points += points
        
    print("="*50)
    main_timer.stop()
    total_points = int(round(total_points))
    print(f"Totals Points Earned: {total_points}/80")
    
    elapsed_time = main_timer.last_elapsed_time
    final_mse = task_vld_scores['Regressor']['MSE']
    final_acc = task_vld_scores['Classifier']['Accuracy']
    return total_points, elapsed_time, final_mse, final_acc

# def set_seeds(seed):
#     np.random.seed(25)
#     random.seed(25)

def reshape_labels(y):
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return y

def get_name(obj):
    try:
        if hasattr(obj, '__name__'):
            return obj.__name__
        else:
            return obj
    except Exception as e:
        return obj
    
# def mse(y, y_hat):
#     # Checks if y or y_hat need to be
#     # reshaped into 2D array
#     y = reshape_labels(y)
#     y_hat = reshape_labels(y_hat)
#     err = y_hat - y 
#     return np.mean(err**2)

# def accuracy(y, y_hat):
#     # Convert y from one-hot encoding back to normal
#     if len(y.shape) > 1 and y.shape[-1] > 1:
#         y = np.argmax(y, axis=1).reshape(-1,1)
#     # Reshape labels and preds to be 2D arrays
#     elif len(y.shape) == 1:
#         y = y.reshape(-1, 1)
#     if len(y_hat.shape) == 1:
#         y_hat = y_hat.reshape(-1, 1)
    
#     return accuracy_score(y, y_hat)
  
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
    
    def fit(self, *args, **kwargs):
        print(f"{self.t1}Training {self.model_name}...")
        print(f"{self.t2}Using hyperparameters: ")
        [print(f"{self.t3}{n} = {get_name(v)}")for n, v in self.model_params.items()]
        try: 
            return self._fit(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while training model for {self.model_name}:"
            catch_and_throw(e, err)
            
    def _fit(self, X, y, metrics=None, pass_y=False):
        if pass_y:
            self.model.fit(X, y)
        else:
             self.model.fit(X)
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

def rubric_regression(mse, max_score=40):
    thresh = 13
    if mse <= thresh:
        score_percent = 100
    elif mse is not None:
        score_percent = (thresh / mse) * 100
        if score_percent < 40:
            score_percent = 40
    else:
        score_percent = 20
    score = max_score * score_percent / 100.0

    return score

def rubric_classification(acc, max_score=40):
    score_percent = 0
    if acc >= 0.90:
        score_percent = 100
    elif acc >= 0.80:
        score_percent = 90
    elif acc >= 0.70:
        score_percent = 80
    elif acc >= 0.60:
        score_percent = 70
    elif acc >= 0.50:
        score_percent = 60
    elif acc >= 0.40:
        score_percent = 55
    elif acc >= 0.30:
        score_percent = 50
    elif acc >= 0.20:
        score_percent = 45
    else:
        score_percent = 20
    score = max_score * score_percent / 100.0 
    return score

if __name__ == "__main__":
    run_eval()


