import os
import joblib
import pyrootutils
from tqdm import tqdm
import numpy as np
from scipy.stats import combine_pvalues
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from statsmodels.distributions.empirical_distribution import ECDF

# from src.utils.functions_library import CauchyCombinationTest, fastSPARDA, gmm_bic_score
# from src.model.components.safetycage import SafetyCage

from safetycage_testing.utils.functions_library import CauchyCombinationTest, fastSPARDA, gmm_bic_score
from safetycage_testing.ABC.safetycage import SafetyCage

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class SPARDACUS(SafetyCage):
    def __init__(self, model_handler, data_handler, **kwargs):
        super(SPARDACUS, self).__init__(model_handler, data_handler, **kwargs)

        self.s_statistic_source = kwargs.get("s_statistic_source")
        self.alpha = kwargs.get("alpha", None)
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers")

        self.classes = data_handler.classes
    @property
    def name(self):
        return "SPARDACUS"

    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Train the spardacus safetycage  using correct and incorrect predictions.

        Args:
            x: Tuple of (x_correct, x_incorrect) input data
            y: Tuple of (y_correct, y_incorrect) labels
        """

        if x is None:
            x, y = self.data_handler.data_train
        if y is None:
            _, y = self.data_handler.data_train
        if y_pred is None:
            y_pred = self.model_handler._get_predictions(x)

        if self.model_handler.use_onehot_encoder:
            mask = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        else:
            mask = y_pred == y
        
        if isinstance(x, dict):
            x_correct = {key: val[mask] for key, val in x.items()}
            x_incorrect = {key: val[~mask] for key, val in x.items()}
        else:
            x_correct = x[mask]
            x_incorrect = x[~mask]
            
        y_correct = y[mask]
        y_incorrect = y[~mask]
        
        # Get layer activations
        layers_activations = {
            "correct": self.model_handler._get_activations(x_correct),
            "incorrect": self.model_handler._get_activations(x_incorrect)
        }
        

        # Initialize parameters dictionary
        selected_layers = self.model_handler.selected_layers
        self.layer_params = {
            layer: {class_index: {} for class_index in self.classes}
            for layer in selected_layers
        }
        # Process each layer and class
        for layer in selected_layers:
            for class_key, class_label in tqdm(self.classes.items()):
                
                # Get class-specific activations
                class_activations_correct = self._get_class_activations(
                    layers_activations["correct"], layer, y_correct, class_key
                )
                class_activations_incorrect = self._get_class_activations(
                    layers_activations["incorrect"], layer, y_incorrect, class_key
                )
                
                # Process layer and class
                self.layer_params[layer][class_label] = self._process_layer_class(
                    class_activations_correct, class_activations_incorrect
                )


    def _get_class_activations(self, layer_activations: dict, layer: str, 
                            y_data: np.ndarray, class_index: int) -> np.ndarray:
        """Extract class-specific activations based on labels."""
        if self.model_handler.use_onehot_encoder:
            return layer_activations[layer][y_data[:, class_index] == 1, :]
        
        return layer_activations[layer][y_data == class_index, :]


    def _process_layer_class(self,class_activations_correct: np.ndarray, class_activations_incorrect: np.ndarray) -> dict:
        """Process activations for a single layer and class."""
        # Run fastSPARDA
        beta_hat, _, _, _ = fastSPARDA(
            X_samples = class_activations_correct, 
            Y_samples = class_activations_incorrect
            )
        
        # Get projected samples
        predicted_samples_correct = np.dot(class_activations_correct, beta_hat)
        predicted_samples_incorrect = np.dot(class_activations_incorrect, beta_hat)
        
        # Fit density estimators
        density_correct = self._fit_gaussian_mixture(predicted_samples_correct)
        density_incorrect = self._fit_gaussian_mixture(predicted_samples_incorrect)
        
        # Compute log PDFs
        pdf_results = self._compute_log_pdfs(density_correct, density_incorrect)
        
        # Initialize statistics
        score_statistic_correct = None
        score_statistic_incorrect = None

        # Compute relevant statistics based on configuration
        if self.s_statistic_source == "correctly":
            score_statistic_correct = pdf_results["ln_pdf_h1_correct"] - pdf_results["ln_pdf_h0_correct"]
            
        if self.s_statistic_source == "incorrectly":
            score_statistic_incorrect = pdf_results["ln_pdf_h1_incorrect"] - pdf_results["ln_pdf_h0_incorrect"]
        
        # Compute ECDFs
        ecdf_correct = ECDF(score_statistic_correct) if score_statistic_correct is not None else None
        ecdf_incorrect = ECDF(score_statistic_incorrect) if score_statistic_incorrect is not None else None
        
        return {
            "ecdf_correct": ecdf_correct,
            "ecdf_incorrect": ecdf_incorrect,
            "beta_hat": beta_hat,
            "density_correct": density_correct,
            "density_incorrect": density_incorrect
        }


    def _fit_gaussian_mixture(self, samples: np.ndarray) -> GaussianMixture:
        """Fit Gaussian Mixture Model using grid search."""
        param_grid = {
            "n_components": range(1, 4),
            "covariance_type": ["full"],
        }
        
        grid_search = GridSearchCV(
            estimator=GaussianMixture(),
            param_grid=param_grid,
            scoring=gmm_bic_score,
            cv=2
        )
        
        grid_search.fit(samples.reshape(-1, 1))
        return grid_search.best_estimator_


    def _compute_log_pdfs(self, density_correct: GaussianMixture, density_incorrect: GaussianMixture, n_samples: int = int(1e6)) -> dict:
        
        """Compute log PDFs for correct and incorrect samples."""
        samples_correct = density_correct.sample(n_samples)[0]
        samples_incorrect = density_incorrect.sample(n_samples)[0]
        
        ln_pdf_h0_correct = density_correct.score_samples(samples_correct)
        ln_pdf_h1_correct = density_incorrect.score_samples(samples_correct)
        ln_pdf_h0_incorrect = density_correct.score_samples(samples_incorrect)
        ln_pdf_h1_incorrect = density_incorrect.score_samples(samples_incorrect)
        
        return {
            "samples_correct": samples_correct,
            "samples_incorrect": samples_incorrect,
            "ln_pdf_h0_correct": ln_pdf_h0_correct,
            "ln_pdf_h1_correct": ln_pdf_h1_correct,
            "ln_pdf_h0_incorrect": ln_pdf_h0_incorrect,
            "ln_pdf_h1_incorrect": ln_pdf_h1_incorrect
        }


    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Tests cage on given data.
        
        Args:
            x: Input features array
            y: Target values array
            
        Returns:
            np.ndarray: Combined p-values. Shape depends on s_statistic_source:
                        vector of global p-values per sample
        """
        
        pvalue = self._compute_statistics(x, y)
        
        return self._combine_layer_pvalues(pvalue, len(y), self.test_type_between_layers)


    def _compute_statistics(self, x, y):
        
        selected_layers = self.model_handler.selected_layers

        num_samples = len(y)
        num_layers = len(selected_layers)

        pvalue = np.full(
            shape = (num_samples, num_layers),
            fill_value = np.inf,
            dtype = np.float64
            )

        # NOTE: my implementation
        # pvalue = {
        #     layer: None for layer in selected_layers
        # }
        # get activation for all predictions
        
        activations = self.model_handler._get_activations(x)
        
        for layer_index, layer in enumerate(selected_layers): # for all layers
            for sample_index, y_sample, in enumerate(y): # for all predictions to be tested
                
                # Compute p-values of each sample per layer using ECDF function
                if self.model_handler.use_onehot_encoder:
                    class_label = self.classes[np.argmax(y_sample)]
                else:
                    class_label = self.classes[y_sample]
                
                ## Get the projection vector beta hat and the actication for the sample
                activation = activations[layer][sample_index]
                beta_hat = self.layer_params[layer][class_label]["beta_hat"]
                
                # Compute observed value with respect to beta_hat_i projection for predicted class y[sample_index]:
                activation_projected = np.dot(activation, beta_hat).reshape(1,-1)
                
                # Get the density functions of correctly and incorrectly predicted samples, for the layer
                density_correct = self.layer_params[layer][class_label]["density_correct"]
                density_incorrect = self.layer_params[layer][class_label]["density_incorrect"]
                
                # compute the s statistic for the sample
                # since -ln(a/b) = ln(b)-ln(a)
                statistic = np.subtract(
                    density_incorrect.score_samples(activation_projected),
                    density_correct.score_samples(activation_projected)
                    )
                
                
                # Get the ECDF functions for the layer
                ecdf_correct = self.layer_params[layer][class_label]["ecdf_correct"]
                ecdf_incorrect = self.layer_params[layer][class_label]["ecdf_incorrect"]
                
                if self.s_statistic_source == "correctly": 
                    # Right-sided test. Small p-value indicates sample is incorrectly classfied                           
                    pvalue[sample_index,layer_index] = 1 - ecdf_correct(statistic)
                    
                elif self.s_statistic_source == "incorrectly":
                    # Left-sided test. Small p-value indicates sample is correctly classified.
                    pvalue[sample_index,layer_index] = ecdf_incorrect(statistic)
                    

                # NOTE My implementation:
                # pvalue_correct = None
                # pvalue_incorrect = None
                
                # if self.s_statistic_source in ["both", "correctly"]: 
                #     # Right-sided test. Small p-value indicates sample is incorrectly classfied
                #     pvalue_correct = 1 - ecdf_correct(statistic)                      
                    
                # elif self.s_statistic_source in ["both", "incorrectly"]:
                #     # Left-sided test. Small p-value indicates sample is correctly classified.
                #     pvalue_incorrect = 1 - ecdf_incorrect(statistic)                      
                    

                #     pvalue[sample_index,layer_index] = {
                #         "correctly": pvalue_correct,
                #         "incorrectly": pvalue_incorrect
                #     }

        
        return pvalue


    def _combine_layer_pvalues(self, pvalues: np.ndarray, y_len: int, test_type: str | None = None) -> np.ndarray:
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
                for i in range(y_len)
            ])
        
        if test_type == 'cauchy':
            return np.array([
                CauchyCombinationTest(
                    p_values = pvalues[i, :],
                    weights = self.cauchy_weights_per_layer
                    )
                for i in range(y_len)
            ])
        
        raise ValueError(f"Unknown test type: {test_type}")


    def flag(self, statistics, alpha=None):
        
        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        
        if self.s_statistic_source == "correctly":
            #if alpha argument to flag() function not none, use this and not the one in config-file
            flags = (statistics <= alpha)
            
        #small p-value indicates sample is correctly classified. Make sure flag = 1 means prediction is deemed to be wrong
        elif self.s_statistic_source == "incorrectly":
            flags = ~(statistics <= alpha)

        return flags

if __name__ == "__main__":
    SPARDACUS(None, None, None)