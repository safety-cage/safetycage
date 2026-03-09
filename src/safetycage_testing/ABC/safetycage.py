from typing import Any
import pyrootutils
from abc import ABC, abstractmethod
import numpy as np
import os
import json
import joblib
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class SafetyCage(ABC):
    def __init__(
        self,
        model_handler,
        data_handler: Any,
        **kwargs
        ) -> None:

        self.model_handler = model_handler
        self.data_handler = data_handler
        
        self.num_classes = data_handler.num_classes
        self.selected_classes = data_handler.classes
        
        self.alpha = None
        
    #Train the parameters of the specified SafetyCage
    @abstractmethod
    def train_cage(self) -> None:
        pass
    
    #Apply the SafetyCage on unseen test samples
    @abstractmethod
    def predict(self, x, y) -> None:
        pass

    #Compute the statistics to evaluate whether each test sample is wrongly predicted
    @abstractmethod
    def _compute_statistics(self, x, y):
        pass

    #Flag predictions as being correct (0) or wrong (1)
    def flag(self, statistics: float | np.ndarray, alpha: float | None = None) -> float | np.ndarray:
        """Flag samples with probability less than or equal (safetycage.leq = True) to alpha as incorrect
        or probability more than or equal (safetycage.leq = False) to alpha as incorrect.
        
        This method identifies samples where the maximum/minimum probability is below/above a
        specified threshold (alpha), marking them as potentially incorrect classifications.
        Requires safetycage.leq to be defined, not None.

        Args:
            statistics (numpy.ndarray): Array of probability values to evaluate
            alpha (float): Threshold value for flagging samples (0 to 1)
        Returns:
            numpy.ndarray: Boolean array where True indicates probabilities below alpha threshold
        """
                
        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        if self.leq:
            flags = statistics <= alpha
        elif not self.leq and self.leq is not None:
            flags = statistics >= alpha
        else:
            raise ValueError("Define safetycage.leq")

        return flags

    def find_best_threshold_flag(self, y_true, y_probs, metric_fn, greater_is_better=True) -> float | np.ndarray:
        """
        Call self.flag() to determine the optimal threshold.
        """
        if greater_is_better:
            compare = lambda x, y: x > y
        else:
            compare = lambda x, y: x < y

        best_metric = -np.inf 
        best_alpha = -np.inf
        for t in np.linspace(min(y_probs), max(y_probs), num=1000):

            flag = self.flag(y_probs, t)

            # flag is true when misclassification occurs
            tps = np.sum(flag & y_true)
            fps = np.sum(flag & (1 - y_true))
            
            total_pos = y_true.sum()
            total_neg = y_true.size - total_pos
            
            fns = total_pos - tps
            tns = total_neg - fps
            
            metric = metric_fn(TP=tps, TN=tns, FP=fps, FN=fns)
            
            if compare(metric, best_metric):
                best_metric = metric
                best_alpha = t
                
        return {
            "alpha_opt": best_alpha,
            "metric_max": best_metric,
        }
    
    def save_cage(self, path):
        """Save the safety cage parameters, alpha and/or layer_params, to a specified folder path
        in a file called parameters.json and/or parameters.joblib.

        Args:
            path (str): Folder path where the safety cage parameters should be saved
        """

        os.makedirs(path, exist_ok=True)

        if getattr(self, "alpha", None) is not None:
            if getattr(self, "layer_params", None) is not None:
                
                parameters_path = os.path.join(path, 'parameters.joblib')
                parameters = {
                    'alpha': self.alpha,
                    'layer_params': self.layer_params
                }
                joblib.dump(parameters, parameters_path)
            
            else:

                parameters_path = os.path.join(path, 'parameters.json')

                with open(parameters_path, 'w') as f:
                    json.dump({'alpha': str(self.alpha)}, f)

        else:
            raise ValueError("alpha is not set.")

    def load_cage(self, path):
        """Load the safety cage parameters, alpha and/or layer_params, from a specified folder path.
        Looks for a parameters.json or parameters.joblib file respectively.

        Args:
            path (str): Path from where the safety cage parameters should be loaded
        """
        joblib_path = os.path.join(path, "parameters.joblib")
        json_path = os.path.join(path, "parameters.json")

        if os.path.isfile(path):
            if path.endswith(".joblib"):
                
                parameters = joblib.load(path)
                self.alpha = parameters["alpha"]
                self.layer_params = parameters["layer_params"]

            elif path.endswith(".json"):

                with open(path, "r") as f:
                    data = json.load(f)
                self.alpha = float(data["alpha"])

            else:
                raise ValueError(f"Unsupported file type: {path}")
            
        elif os.path.exists(joblib_path):

            parameters = joblib.load(joblib_path)
            
            # Restore parameters
            self.alpha = parameters['alpha']
            self.layer_params = parameters['layer_params']

        elif os.path.exists(json_path):

            with open(json_path, 'r') as f:
                data = json.load(f)
                self.alpha = float(data['alpha'])
        else:
            raise FileNotFoundError(f"Safety cage parameters file not found at {path}")

if __name__ == "__main__":
    SafetyCage(None, None, None)