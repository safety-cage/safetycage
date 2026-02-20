from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Any, Dict
import numpy as np
import omegaconf

class ModelHandler(ABC):
    def __init__(
        self,
        selected_layers:Union[str, List[str]],
        use_onehot_encoder:bool,
        model:Any,
        **kwargs
        ):
        super(ModelHandler, self).__init__()
        
        self.model = model
        self.use_onehot_encoder = use_onehot_encoder

        # Handle different types of selected_layers input
        if isinstance(selected_layers, str):
            self.selected_layers = [selected_layers]
        elif isinstance(selected_layers, (list, omegaconf.listconfig.ListConfig)):
            if all(isinstance(layer, str) for layer in selected_layers):
                self.selected_layers = list(selected_layers)  # Convert to regular list if it's ListConfig
            else:
                raise ValueError("All elements in selected_layers must be strings")
        else:
            raise ValueError(f"selected_layers must be a string or list of strings, got {type(selected_layers)}")


    @abstractmethod
    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x."""
        raise NotImplementedError("Implement based on your model architecture")
    
    
    @abstractmethod
    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        # This method should be implemented based on your specific model architecture
        raise NotImplementedError("Implement based on your model architecture")


    @abstractmethod
    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        raise NotImplementedError("Implement based on your model architecture")


    @abstractmethod
    def _calc_model_shape(self) -> Dict[str,int]:
        """
        Get the shape of each layer in the model.
        Returns: List of integers representing the number of neurons in each layer
        """
        raise NotImplementedError("Implement based on your model architecture")
    
if __name__ == '__main__':
    #Quick test to confirm it builds
    model_handler = ModelHandler()
