import numpy as np

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from ...model.components.safetycage import SafetyCage
from safetycage_testing.ABC.safetycage import SafetyCage

import json
import os

class MSP(SafetyCage):
    def __init__(self, model_handler, data_handler, **kwargs):
        super(MSP, self).__init__(model_handler, data_handler, **kwargs)
        self.leq = True
    """
    Maximum Softmax Probability (MSP) Safety Cage.
    
    This class implements a simple safety cage based on maximum softmax probability thresholding
    for detecting uncertain predictions in neural network classifiers. The approach flags predictions
    as potentially incorrect when the maximum softmax probability falls below a specified threshold.
    
    The method is based on the research presented in:
    Hendrycks, D., & Gimpel, K. (2016). A Baseline for Detecting Misclassified and 
    Out-of-Distribution Examples in Neural Networks. arXiv:1610.02136.
    
    Attributes:
        train_cage_data (tuple): Stores training data (x,y) used to train the safety cage
        model_handler: Reference to model handler object for making predictions
    
    Methods:
        train_cage: Stores training data for the safety cage
        predict: Computes statistical metrics on input data
        _compute_statistics: Calculates maximum softmax probabilities
        flag: Identifies uncertain predictions based on probability threshold
    """
    @property
    def name(self):
        return "MSP"
    
    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        pass

    def predict(self, x, y) -> None:
        
        statistics = self._compute_statistics(x)

        return statistics
    
    def _compute_statistics(self, x):
        """
        Compute statistical metrics based on model predictions.
        This method processes input data through the model to get softmax probabilities
        and returns the maximum probability for each sample.
        Args:
            x (numpy.ndarray): Input data to be processed by the model
            y (numpy.ndarray): Target labels (not used in current implementation)
        Returns:
            numpy.ndarray: Array of maximum probabilities for each input sample
        """
        
        # Get softmax probabilities from model
        probabilities = self.model_handler._get_probabilities(x)
        
        # Get maximum probability for each sample
        max_probabilities = np.max(probabilities, axis=1)
        
        return max_probabilities

if __name__ == "__main__":
    MSP(None, None)