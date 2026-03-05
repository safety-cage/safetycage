from typing import Any
import pyrootutils
from abc import ABC, abstractmethod
import numpy as np
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
    @abstractmethod
    def flag(self, statistics, alpha = None):
        pass

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
    
    @abstractmethod
    def save_cage(self, parameters, path):
        pass
    
    @abstractmethod
    def load_cage(self, path):
        pass

if __name__ == "__main__":
    SafetyCage(None, None, None)