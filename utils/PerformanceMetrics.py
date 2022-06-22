# Built-in libraries
import math
import numpy as np
import pandas                           as pd

# Sklearn
#
from sklearn                 import metrics



def smape(A, F):
    try:
        return ( 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) ) )
    except:
        return (np.NaN)
    
def rmse(A, F):
    try:
        return math.sqrt(metrics.mean_squared_error(A, F))
    except:
        return (np.NaN)


def RegressionEvaluation( Prices ):
    '''
    Parameters
    ----------
    Y : TYPE
        Real prices.
    Pred : TYPE
        Predicted prices.
    Returns
    -------
    MAE : TYPE
        Mean Absolute Error.
    RMSE : TYPE
        Root Mean Square Error.
    MAPE : TYPE
        Mean Absolute Percentage Error.
    R2   : TYPE
        R2 correlation
    '''
    
    SeriesName = Prices.columns[0]
    Prediction = Prices.columns[1]
    
    Y    = Prices[SeriesName].to_numpy()
    Pred = Prices[Prediction].to_numpy()
    
    
    
    MAE   = metrics.mean_absolute_error(Y, Pred)
    RMSE  = math.sqrt(metrics.mean_squared_error(Y, Pred))
    try:
        MAPE  = np.mean(np.abs((Y - Pred) / Y)) * 100.0
    except:
        MAPE  = np.NaN
        
    SMAPE = smape(Y, Pred)
    R2    = metrics.r2_score(Y, Pred)
    

        
    return (MAE, RMSE, MAPE, SMAPE, R2)