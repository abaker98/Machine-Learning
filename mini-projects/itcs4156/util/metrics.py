from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def reshape_labels(y):
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return y

def error(y, y_hat):
    # Checks if y or y_hat need to be
    # reshaped into 2D array
    y = reshape_labels(y=y)
    y_hat = reshape_labels(y=y_hat)
    
    return y_hat - y 

def sse(y, y_hat):
    err = error(y=y, y_hat=y_hat)
    return np.sum(err**2)

def mse(y, y_hat):
    err = error(y=y, y_hat=y_hat)
    return np.mean(err**2)

def rmse(y, y_hat):
    return np.sqrt(mse(y=y, y_hat=y_hat))

def performance_measures(y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray]:
    sse_ = sse(y=y, y_hat=y_hat)
    mse_ = mse(y=y, y_hat=y_hat)
    rmse_ = rmse(y=y, y_hat=y_hat)
    return sse_, mse_, rmse_

def nll(y, y_hat, epsilon=1e-5):
    loss = y * np.log(y_hat+epsilon)
    cost = -np.sum(loss)
    return cost

def mean_nll(y, y_hat, epsilon=1e-5):
    return nll(y=y, y_hat=y_hat, epsilon=epsilon) / len(y)

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
  
def ppv(tp, fp):
    return tp / (tp + fp)

def tpr(tp, fn):
    return tp / (tp + fn)

def tnr(tn, fp):
    return tn / (tn + fp)

def compute_scores(y, y_hat, class_names=None):
    def print_scores(tn, fn, fp, tp):

        print(f"\tPPV ratio tp/(tp+fp): {tp}/{tp+fp}")
        print(f"\tPPV (precision): {ppv(tp=tp, fp=fp) }\n")

        print(f"\tTPR ratio tp/(tp+fn): {tp}/{tp+fn}")
        print(f"\tTPR (recall/sensitivity): {tpr(tp=tp, fn=fn)}\n")

        print(f"\tTNR ratio tn/(tn+fp): {tn}/{tn+fp}")
        print(f"\tTNR (specificity): {tnr(tn=tn, fp=fp)}")
    
    if class_names is None:
        class_names = {}
    
    # Convert y from one-hot encoding back to normal
    if len(y.shape) > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=1).reshape(-1,1)
    
    cm = confusion_matrix(y_true=y, y_pred=y_hat)
    
    # Computing multi-class classification tp, fn, tp, tn
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    ppv_ = ppv(tp=tp, fp=fp) 
    tpr_ = tpr(tp=tp, fn=fn)
    tnr_ = tnr(tn=tn, fp=fp)
    
    class_labels = np.unique(y)
    
    if len(class_labels) == 2:
        class_name = class_names.get(class_labels[-1], class_labels[-1])
        print(f"Scores for binary problem: positive label is {class_name}")
        print_scores(tn[-1], fn[-1], fp[-1], tp[-1])
    else:
        for i, label in enumerate(class_labels):
            class_name = class_names.get(label, label)
            print(f"Scores for class {class_name}")
            print_scores(tn[i], fn[i], fp[i], tp[i])