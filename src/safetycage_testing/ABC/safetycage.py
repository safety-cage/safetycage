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
    
    @abstractmethod
    def save_cage(self, parameters, path):
        pass
    
    @abstractmethod
    def load_cage(self, path):
        pass

if __name__ == "__main__":
    SafetyCage(None, None, None)