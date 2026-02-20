import os
import joblib
import numpy as np
from numpy import linalg
from scipy.stats import chi2, norm, f, combine_pvalues
from statsmodels.distributions.empirical_distribution import ECDF

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ..utils.functions_library import CauchyCombinationTest

from ..ABC.safetycage import SafetyCage

class Mahalanobis(SafetyCage):
    def __init__(self, model_handler, data_handler,**kwargs):
        super(Mahalanobis, self).__init__(model_handler, data_handler, **kwargs)
        
        self.empirical = kwargs.get("empirical")
        self.use_preactivations = kwargs.get("use_preactivations")
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers")
        self.test_type_within_layer = kwargs.get("test_type_within_layer")
        
        self.selected_layers = self.model_handler.selected_layers
        self.last_layer = self.model_handler.last_layer
        self.classes = data_handler.classes

        if self.model_handler.use_onehot_encoder:
            self.index_function = lambda x: self.classes.index(np.argmax(x))
        else:
            self.index_function = lambda x: self.classes.index(x)
            
        self.test_type_fn_dict = {
            "chi2": self.chi2_statistic,
            "t2": self.t2_statistic,
            "mahalanobis": self.mahalanobis_statistic
        }
        
    @property
    def name(self):
        return "mahalanobis"


        
    def train_cage(self, x, y, y_pred) -> None:
        
        # mahalanobis distance is used to compute the p-value

        if self.model_handler.use_onehot_encoder:
            mask = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        else:
            mask = y_pred == y

        x_correct = x[mask]
        y_correct = y[mask]


        layer_activations =  self.model_handler._get_pre_activations(x_correct)
        
        self.layer_params = {
            layer: {class_index: {} for class_index in self.classes}
            for layer in self.selected_layers
        }
        
        # Process each layer and class
        for _class in self.selected_classes:
            class_index = self.classes.index(_class)
                
            if self.model_handler.use_onehot_encoder:
                num_observations = np.sum(y_correct[:, class_index] == 1)
            else:
                num_observations = np.sum(y_correct == _class)
            
            for layer in self.selected_layers:
                
                class_activations = self._get_class_activations(
                    layer_activations = layer_activations,
                    layer = layer,
                    y_data = y_correct,
                    class_index = class_index
                    )
                
                self.layer_params[layer][class_index] = self.compute_empirical_distribution(
                    class_activations = class_activations,
                    empirical = self.empirical,
                    num_observations = num_observations
                    )



    def _get_class_activations(self, layer_activations: dict, layer: str, 
                            y_data: np.ndarray, class_index: int) -> np.ndarray:
        """Extract class-specific activations based on labels."""
        if self.model_handler.use_onehot_encoder:
            return layer_activations[layer][y_data[:, class_index] == 1, :]
        
        return layer_activations[layer][y_data == class_index, :]
    
    
    def predict(self, x, y) -> None:
        """Compute p-values for input data using Mahalanobis distance.
        
        Computes p-values per layer for each sample and combines them into a global p-value.
        Note: For activation values, nodes that are never activated need special handling.
        
        Args:
            x: Input data samples
            y: True labels
            
        Returns:
            combined_pvalue: Global p-value per sample
        """
        
        # Calculate p-values for each layer
        pvalue = self._compute_statistics(x, y)

        # Combine p-values across layers into global p-values per sample
        combined_pvalue = self._combine_layer_pvalues(
            pvalues = pvalue, 
            num_samples = len(y), 
            test_type = self.test_type_between_layers
        )
        
        return combined_pvalue


    def _combine_layer_pvalues(self, pvalues: np.ndarray, num_samples: int, test_type: str | None = None) -> np.ndarray:
        """Combine p-values across layers using the specified method."""
        num_layers = pvalues.shape[1]
        
            
        if test_type is None and num_layers > 1:
            raise ValueError("test_type_between_layers cannot be None when combining p-values between several layers")
            
        if num_layers == 1:
            return pvalues[:, 0]
        
        if test_type == 'fisher':
            return np.array([
                combine_pvalues(
                    pvalues = pvalues[i, :],
                    method = "fisher"
                    )[1]
                for i in range(num_samples)
            ])
        
        if test_type == 'cauchy':
            return np.array([
                CauchyCombinationTest(
                    p_values = pvalues[i, :],
                    weights = self.cauchy_weights_per_layer
                    )
                for i in range(num_samples)
            ])
        
        raise ValueError(f"Unknown test type: {test_type}")
    

    def _compute_statistics(self, x, y):

        num_samples = len(y)
        num_layers = len(self.selected_layers)
        
        pvalue = np.full(
            shape = (num_samples, num_layers),
            fill_value = np.inf,
            dtype  = np.float64
        )
                
        activations = self.model_handler._get_pre_activations(x)

        test_type = self.test_type_fn_dict[self.test_type_within_layer]

        # Compute p-value per layer using the mahalanobis distance. See https://stats.stackexchange.com/questions/416198/calculate-p-value-of-multivariate-normal-distribution
        
        # for all predictions to be tested
        for sample_index, y_sample  in enumerate(y):
            
            # get the class index
            class_index = self.index_function(y_sample)
            
            # for all layers ...
            for layer_index, layer in enumerate(self.selected_layers):
                
                # get the activations of the sample for the layer
                activation = activations[layer][sample_index]
                
                # If we are not at the last layer:
                if layer != self.last_layer:
                    pvalue[sample_index, layer_index] = test_type(activation, class_index, layer)

                else:
                    
                    # Multivariate approach using chi2
                    if self.model_handler.use_onehot_encoder: 
                        pvalue[sample_index, layer_index] = self.chi2_statistic(activation, class_index, layer)
                    
                    # Assume univariate normal distribution, and do a two sided-test:
                    else:
                        pvalue[sample_index, layer_index] = self.two_sided_test(activation, class_index, layer)

        # return p-value
        return(pvalue)


    
    def chi2_statistic(self, activation, class_index, layer):
        # Asymptotic assumption => chi2-distribution

        mean = self.layer_params[layer][class_index]["mean"]
        variance = self.layer_params[layer][class_index]["variance"]
        
        activation_centered = (activation - mean)[..., np.newaxis]
        
        inv_var_mean = linalg.solve(variance, activation_centered)
        
        distance = np.matmul(activation_centered.T, inv_var_mean).item()
        # distance = np.matmul(inv_var_mean.T, activation_centered).item()
        
        result = chi2.sf(distance, df=len(activation))
        
        return result
    
    def t2_statistic(self, activation, class_index, layer):
        # Using exact distribution, the Hotelling's T^2 distribution:

        # number of observations for particular class during training
        n = self.layer_params[layer][class_index]["ECDF"]
        
        # The dimension of the random vector oif layer 
        p = np.shape(activation)[0]
        
        part_1 = (activation-self.layer_params[layer][class_index][0])[np.newaxis, ...]
        part_2 = linalg.solve(
            self.layer_params[layer][class_index][1],
            (activation-self.layer_params[layer][class_index][0])[..., np.newaxis]
            )
        
        f_obs = ((n-p)/(p*(n-1))) * np.matmul(part_1, part_2)[0][0]
        
        result = f.sf(f_obs, dfn=p, dfd=n-p)
        return result
    
    def mahalanobis_statistic(self, activation, class_index, layer):


        part_1 = (activation-self.layer_params[layer][class_index]["mean"])[np.newaxis, ...]

        part_2 = linalg.solve(
            a = self.layer_params[layer][class_index]["variance"],
            b = (activation-self.layer_params[layer][class_index]["mean"])[..., np.newaxis]
            )
        
        product = np.matmul(part_1, part_2)[0][0]

        # Compute p-value using empirical distribution of Mahalanobis distance of correcly classified samples for the particular class:
        result = 1 - self.layer_params[layer][class_index]["ECDF"](product)
        
        return result
    
    def two_sided_test(self, activation, class_index, layer):

        mean = self.layer_params[layer][class_index]["mean"]
        variance = self.layer_params[layer][class_index]["variance"]

        # Calculate values for upper and lower tail probabilities
        upper_bound = activation if activation > mean else 2*mean - activation
        lower_bound = 2 * mean - activation if activation > mean else activation

        # Calculate p-value using the same formula in both cases
        upper_tail_prob = norm.sf(
            x = upper_bound,
            loc = mean,
            scale = variance
            )
        
        lower_tail_prob = norm.cdf(
            x = lower_bound,
            loc = mean,
            scale = variance
            )
        
        return upper_tail_prob + lower_tail_prob
    
    def compute_empirical_distribution(self, class_activations:np.ndarray, empirical:bool, num_observations:int):

        # compute activation statistic moments 
        sample_mean = np.mean(class_activations, axis = 0)
        sample_var = np.cov(class_activations, rowvar=False) if len(class_activations) > 1 else np.zeros_like(class_activations)

        if not empirical:
            empirical_distribution_statistics = {
                "mean": sample_mean,
                "variance": sample_var,
                "num_observations": num_observations
            }
        
        else:
            mahalanobis_distances = []
            for activation in class_activations:
                
                # Compute difference between activation and sample mean
                diff = activation - sample_mean
                
                # Reshape diff to match the dimensions of sample_var
                diff_reshaped = diff[np.newaxis, ...]
                
                # Solve the linear system of equations
                solved = linalg.solve(sample_var, diff[..., np.newaxis])
                
                # Compute Mahalanobis distance
                distance = np.matmul(diff_reshaped, solved)[0][0]
                mahalanobis_distances.append(distance)
            
            # Compute empirical distribution
            empirical_distribution_statistics = {
                "mean": sample_mean,
                "variance": sample_var,
                "ECDF": ECDF(mahalanobis_distances)
            }
            
        return empirical_distribution_statistics

    def flag(self, statistics: float | np.ndarray, alpha: float | None = None) -> float | np.ndarray:
        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        flags = statistics <= alpha

        return flags

    def save_cage(self, path):

        # Create a dictionary containing all parameters including alpha
        parameters = {
            'alpha': self.alpha,
            'layer_params': self.layer_params
        }
        
        # Save all parameters using joblib
        filepath = os.path.join(path, 'parameters.joblib')
        joblib.dump(parameters, filepath)

    def load_cage(self, path):
        parameter_path = os.path.join(path, "parameters.json")

        parameters = joblib.load(parameter_path)
        
        # Restore parameters
        self.alpha = parameters['alpha']
        self.layer_params = parameters['layer_params']

        
if __name__ == "__main__":
    Mahalanobis(None, None, None)