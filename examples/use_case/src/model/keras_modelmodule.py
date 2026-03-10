from typing import List, Any, Dict
import numpy as np

import keras.backend as K
import tensorflow as tf

from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(search_from=Path(__file__), indicator=[".project-root"], pythonpath=True)

from safetycage_testing.ABC.modelmodule import ModelModule

class KerasModelModule(ModelModule):
    def __init__(
        self,
        selected_layers: List[str],
        use_onehot_encoder: bool,
        model: Any,
        **kwargs
        ):
        super(KerasModelModule, self).__init__(
            selected_layers,
            use_onehot_encoder,
            model,
            **kwargs
            )
        """
        Initialize ModelModule with pre-loaded dataset and model.
        ModelModule needs to calculate the following:
            - pre_activations of any given layer
            - activations of any given layer
            - model_shape:
                list that contains the total number of nodes/neurons in each layer of the neural network
            - selected_layers_list:
                specify which layers of the neural network should be analyzed.
        Args:
            dataset: Tuple of ((x_train, y_train), (x_test, y_test))
            model: Pre-trained model instance
            selected_layers: List of layer indices to analyze
        """
        self.model_shape = self._calc_model_shape()
        
        # Convert single layer string to list for consistent handling
        
        self.layers = {name: self.model.get_layer(name=name) for name in self.selected_layers}
        self.last_layer = kwargs.get("last_layer", None)
    
    
    def _get_batched_predictions(
        self,
        dataset: tf.data.Dataset
    ) -> List[np.ndarray]:
        """Get predictions for the entire dataset in batches."""
        predictions = []
        for x_batch, _ in dataset:
            batch_predictions = self._get_predictions(x_batch)
            predictions.append(batch_predictions)
        return np.concatenate(predictions) # Return array

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input data"""
        y = self.model.predict(x)
        predicted_classes = np.argmax(y, axis=1)
        if self.use_onehot_encoder:
            # Convert to one-hot encoding
            onehot = np.zeros((predicted_classes.size, y.shape[1]))
            onehot[np.arange(predicted_classes.size), predicted_classes] = 1
            return onehot
        else:
            return predicted_classes


    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x."""
        
        y = self.model.predict(x)
        
        return y

    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        
        activations = {}
        
        for layer_name, layer in self.layers.items():
            layer_output = layer.output
            get_activations = K.function(self.model.input, layer_output)
            activations[layer_name] = get_activations(x)
        
        return activations
    
    def _get_batched_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        
        activations = {}
        
        for layer_name, layer in self.layers.items():
            layer_output = layer.output
            get_activations = K.function(self.model.input, layer_output)
            activations[layer_name] = get_activations(x)
        
        return activations

    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        
        # Get all layer activations first
        model_input = self.model.input
        all_layer_inputs = {layer_name: None for layer_name in self.layers.items()}
        
        for layer_name, layer in self.layers.items():
            layer_input = layer.input
            get_previous_activation = K.function(model_input, layer_input)
            previous_activation = get_previous_activation(x)
            all_layer_inputs[layer_name] = previous_activation
            
        # Calculate pre-activations using weights and biases
        pre_activations = {}
        for layer_name, layer in self.layers.items():
            pre_activations[layer_name] = np.dot(
                all_layer_inputs[layer_name],
                layer.kernel.numpy()
            ) + layer.bias.numpy()
        
        return pre_activations
    
    # def _get_neuron_values(self, x: np.ndarray) -> np.ndarray:
        
            
    def _calc_model_shape(self) -> Dict[str,int]:
        """
        Returns a dictionary mapping layer names to their output shapes.
        
        Parameters:
        model: Keras model
        
        Returns:
        dict: Keys are layer names, values are shape tuples
        """
        shape_dict = {}
        for layer in self.model.layers:
            shape_dict[layer.name] = layer.output_shape[1]
        return shape_dict
        
if __name__ == '__main__':
    model_module = KerasModelModule()