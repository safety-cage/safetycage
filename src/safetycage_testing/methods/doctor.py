import os
import json
import numpy as np
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ..ABC.safetycage import SafetyCage

class DOCTOR(SafetyCage):
    def __init__(self, model_handler, data_handler, **kwargs):
        super(DOCTOR, self).__init__(model_handler, data_handler, **kwargs)
        self.method = kwargs.get("method")

    """
    DOCTOR  https://arxiv.org/abs/1610.02136
    """
    
    @property
    def name(self):
        return "DOCTOR"
    
    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Compute the estimated probability of making a wrong prediction.
        
        Args:
            x: Input data
            y: Tuple containing (correct_predictions, incorrect_predictions)
        """

        if y is None:
            x, y = self.data_handler.data_train
        if y_pred is None:
            y_pred = self.model_handler._get_predictions(x)

        num_incorrect = (y != y_pred).sum()
        total_samples = len(y)
        
        # Calculate empirical error probability
        self.PE_1 = num_incorrect / total_samples


    def predict(self, x, y) -> None:
        statistics =  self._compute_statistics(x, y)

        return statistics
    
    def _compute_statistics(self, x, y):
        """
        Compute uncertainty statistics based on softmax probabilities.
        
        Args:
            x: Input data
            y: Ground truth labels
            
        Returns:
            float: Uncertainty score Pe(x)/(1-Pe(x))
            
        Raises:
            ValueError: If method is not 'max' or 'sum'
        """
        # Get prediction probabilities from model
        probs = self.model_handler._get_probabilities(x)
        
        # Calculate error probability based on selected method
        if self.method == "max":
            # Use maximum probability
            error_prob = 1 - np.max(probs, axis=1)
        elif self.method == "sum":
            # Use sum of squared probabilities
            error_prob = 1 - np.sum(probs**2, axis=1) 
        else:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'max' or 'sum'")

        # Return uncertainty ratio
        return error_prob / (1 - error_prob)


    def flag(self, statistics: float | np.ndarray, alpha: float | None = None) -> float | np.ndarray:
        """
        Raises a flag if the statistics are higher or equal to a given threshold.
        Args
            statistics : float or array-like
                The odds or probability values to check against the threshold
            alpha : float
                The threshold value to compare against
        Returns:
            flags : bool or array-like of bool
                True where statistics >= alpha, False otherwise
        """
                # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        flags = statistics >= alpha
        
        return flags

    

    def save_cage(self,path):
        """Save the safety cage parameters to a specified path.

        Args:
            path (str): Path where the safety cage parameters should be saved
        """
        if self.alpha:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parameters_path = os.path.join(path, 'parameters.json')
            
            # Save alpha parameter to a JSON file
            with open(parameters_path, 'w') as f:
                json.dump({'alpha': str(self.alpha)}, f)
                
    def load_cage(self, path):
        """Load the safety cage parameters from a specified path.
        
        Args:
            path (str): Path from where the safety cage parameters should be loaded
        """
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.alpha = float(data['alpha'])
        else:
            raise FileNotFoundError(f"Safety cage parameters file not found at {path}")

if __name__ == "__main__":
    DOCTOR(None, None)