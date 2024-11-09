import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np
from bias_correction import BiasCorrection, XBiasCorrection
from torch import nn
import torch
# Read the text file into a DataFrame
# df = pd.read_csv('MATH3133_mv/all_vars.txt', delimiter='\s+')
# df.columns = ['date', 'name', 'lat', 'lon', 'value']

df = pd.read_csv('data/raw/221212.csv')

class BiasCorrector:
    def __init__(self, train_input, ground_truth=None):
        # Input data is (n,T) array where n is the number of datasets and T is the number of time steps
        # Ground truth is (T,) array
        self.train_input = train_input
        self.ground_truth = ground_truth
        self.criterion = nn.MSELoss()

    def train(self, *args, **kwargs):
        # Make the model
        pass
    
    def predict(self, *args, **kwargs):
        # Evaluate the model
        raise NotImplementedError
    
    def evaluate(self, metric="rmse"):
        # Evaluate the model using the given metric
        if metric == "rmse":
            return self.criterion(torch.from_numpy(self.predict(self.train_input)), torch.from_numpy(self.ground_truth)).numpy()

    
##
class QuantileMapping(BiasCorrector):
    def __init__(self, train_input, ground_truth):
        super().__init__(train_input, ground_truth)

        self.ground_truth = pd.Series(ground_truth.flatten())
        self.train_input = pd.Series(train_input.flatten())
    
    def train(self, epoch=None):
        pass
    
    def predict(self, eval_data):
        eval_data = pd.Series(eval_data.flatten())
        return np.array(BiasCorrection(self.ground_truth, self.train_input, eval_data).correct("basic_quantile")).reshape(-1, 1)
    