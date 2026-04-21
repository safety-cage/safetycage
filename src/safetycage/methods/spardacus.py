import os
import joblib
from tqdm import tqdm
import numpy as np
from scipy.stats import combine_pvalues
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

from ..utils.functions_library import CauchyCombinationTest, fastSPARDA, gmm_bic_score
from ..safetycage import SafetyCage

class SPARDACUS(SafetyCage):
    """
    SPARDACUS Safety Cage Method.

    The SPARDACUS safety cage detects misclassified samples by comparing how a sample’s 
    internal neural network activations differ from correctly and incorrectly classified 
    training samples.

    The method models this by learning a projection that separates correct and 
    incorrect activations for each class and layer. During prediction after training,
    it projects a sample’s activation and evaluates how likely it is under both the 
    correct and incorrect distributions. This comparison is then converted into a 
    p-value using fitted density estimators and ECDFs on the likelihood values.

    The smaller the p-value, the more likely the sample is to be misclassified. When 
    multiple layers are used, a global p-value is found by combining the each layer's 
    p-value using either Fisher’s method or the Cauchy combination test. The optimal 
    threshold to compare the resulting p-values is given to the alpha attribute.

    NOTE: This method **only works for neural network models**, as it relies
    on intermediate layer activations and learned representations.

    See the below research paper for a thorough explanation of the SPARDACUS method.

    **Reference:**
        P. V. Johnsen and F. Remonato. “SPARDACUS SafetyCage: A new misclassification detector”.
        https://proceedings.mlr.press/v265/johnsen25a.html

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        classes (dict): Mapping of class indices to class labels.
        layer_params (dict): Stores parameters including such as projection vectors, density 
        estimators, and ECDFs for each layer and class from training.
        unreliable_classes (set): Set of class labels for which density estimation
            failed or was unreliable due to insufficient amount of correct/incorrect predictions.

        s_statistic_source (str): Determines which distribution is used to
            compute p-values ("correctly" or "incorrectly").
        alpha (float | None): Significance threshold used for flagging.
        cauchy_weights_per_layer (list): Weights used for Cauchy combination test.
        test_type_between_layers (str): Method for combining p-values across layers.
        minimum_sample_size (int): Minimum number of samples required to fit
            Gaussian Mixture Models.
    """
    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the SPARDACUS safety cage.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            s_statistic_source (str): Source used to compute p-values ("correctly" or "incorrectly").
            alpha (float): Significance threshold for flagging samples.
            test_type_between_layers (str): Method for combining p-values ("fisher" or "cauchy").
            cauchy_weights_per_layer (list[float]): Weights for the Cauchy combination test.
            minimum_sample_size (int, optional): Minimum number of samples required to fit density models. (default: 10)
        """
        super(SPARDACUS, self).__init__(model_module, data_module, **kwargs)

        self.s_statistic_source = kwargs.get("s_statistic_source")
        self.alpha = kwargs.get("alpha", None)
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers")

        # For the Gaussian Mixture Model fitting. Must be at least 3 based on current implementation (see _fit_gaussian_mixture). Default value is 10.
        if "minimum_sample_size" in kwargs and kwargs["minimum_sample_size"] < 3:
            raise ValueError(f"Minimum_sample_size must be at least 3. Provided: {kwargs['minimum_sample_size']}")
        self.minimum_sample_size = kwargs.get("minimum_sample_size", 10)

        self.classes = data_module.classes
        self.unreliable_classes = set()
    
    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "SPARDACUS"

    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Train the SPARDACUS safety cage.

        Separates training samples into correctly and incorrectly classified groups and
        computes parameters for each layer and class, which are later used to evaluate 
        new samples during prediction.

        Args:
            x: Tuple of (x_correct, x_incorrect) input data
            y: Tuple of (y_correct, y_incorrect) labels
            y_pred: Model predictions
        """
        
        if x is None:
            x, y = self.data_module.data_train
        if y is None:
            _, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        if self.model_module.use_onehot_encoder:
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
            "correct": self.model_module._get_activations(x_correct),
            "incorrect": self.model_module._get_activations(x_incorrect)
        }

        # Initialize parameters dictionary
        selected_layers = self.model_module.selected_layers
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
                    class_activations_correct, class_activations_incorrect, class_label
                )


    def _get_class_activations(self, layer_activations: dict, layer: str, 
                            y_data: np.ndarray, class_index: int) -> np.ndarray:
        """
        Extract activations for a specific class (given by class_index) at a given layer.
        Supports both one-hot encoded and integer labels.

        Returns:
            numpy.ndarray: Activations for the specified class and layer
        """
        if self.model_module.use_onehot_encoder:
            return layer_activations[layer][y_data[:, class_index] == 1, :]
        
        return layer_activations[layer][y_data == class_index, :]


    def _process_layer_class(self, class_activations_correct: np.ndarray, class_activations_incorrect: np.ndarray, class_label: str) -> dict:
        """
        Process activations for a single layer and class.

        Learns a projection that separates correctly and incorrectly classified samples,
        fits density models to the projected values, and prepares statistics used to compute p-values.

        Args:
            class_activations_correct (numpy.ndarray): Activations of correctly classified samples
            class_activations_incorrect (numpy.ndarray): Activations of incorrectly classified samples
            class_label (str): Class label

        Returns:
            dict: Dictionary containing projection vectors, density models, and ECDFs for the class and layer
        """
        # Double check if both class_activations_correct and class_activations_incorrect have a positive number of values
        if len(class_activations_incorrect) == 0 or len(class_activations_correct) == 0:
            warnings.warn(f"No incorrect and/or correct samples for class \"{class_label}\" exist in layer activations. This class will be flagged as unreliable and the results "
                          "for this class are unreliable. We recommend using a different safetycage method or ensuring some incorrect and/or correct samples exist in this class.")
            self.unreliable_classes.add(class_label)
            return {
                "ecdf_correct": None,
                "ecdf_incorrect": None,
                "beta_hat": None,
                "density_correct": None,
                "density_incorrect": None
            }
        
        # Run fastSPARDA
        beta_hat, _, _, _ = fastSPARDA(
            X_samples = class_activations_correct, 
            Y_samples = class_activations_incorrect
            )
        
        # Get projected samples
        predicted_samples_correct = np.dot(class_activations_correct, beta_hat)
        predicted_samples_incorrect = np.dot(class_activations_incorrect, beta_hat)
        
        # Fit density estimators
        density_correct = self._fit_gaussian_mixture(predicted_samples_correct, "correct", class_label)
        density_incorrect = self._fit_gaussian_mixture(predicted_samples_incorrect, "incorrect", class_label)
        
        # Check if fit_gaussian_mixture failed
        if density_correct == None or density_incorrect == None:
            # Assume warning messages and adding to self.unreliable_classes was taken care of in _fit_gaussian_mixture
            return {
                "ecdf_correct": None,
                "ecdf_incorrect": None,
                "beta_hat": None,
                "density_correct": None,
                "density_incorrect": None
            }

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


    def _fit_gaussian_mixture(self, samples: np.ndarray, correctness: str, class_label: str) -> GaussianMixture:
        """
        Notes:
        - Throws an error if there are not enough samples to fit the model, but catches this error and throws a warning instead, returning the best estimator found by GridSearchCV even if it was not properly fitted.

        Fit a Gaussian Mixture Model to the given samples using GridSearchCV for hyperparameter tuning. 
        Use cross-validation with 2-fold CV to select the number of components (1-3) and takes the BIC
        average score of all folds to select the best model. 
        
        **WARNING:** In cases where there are too few samples (often by too few incorrect samples for 
        well-performing models), the Gaussian Mixture Model may fail to fit properly (since it does not 
        make sense to fit a gaussian to too few samples). In such cases, we recommend using a different 
        safetycage method or ensuring enough incorrect and incorrect samples exist for each class.

        For most warning cases the error/warning will be handled below, and a clear warning statement is 
        sent to output. In cases where there are many warnings, it is often a result of sklearn's 
        GridSearchCV struggling to fit.

        Consider the following warning cases:
            - The provided number of samples is less than or minimum_sample_size (which itself has a minimum
            value of 3 since we consider at most 3 components).
            - All Gaussian fits fail (all 6, since there are 3 components to try and 2-fold CV). This is 
            captured as an error and throws a warning. Gaussian fitting does not occur, the given class
            is saved to self.unreliable_classes, and the method returns None.
            - Not all fits failed, but all BIC scores are NaN values (meaning for all component options, 
            at least 1 NaN value occured in the 2-fold CV). The given class is saved to self.unreliable_classes.
            It returns the best estimator found by GridSearchCV, but the found parameter is simply the first one
            available.
            - Not all fits failed, but some BIC scores are NaN values. A warning is thrown that model selection 
            may be unreliable, but the best estimator found by GridSearchCV is returned.

        Args:
            samples (numpy.ndarray): Projected samples
            correctness (str): whethering we are fitting predictions that were "correct" or "incorrect"
            class_label (str): Class label

        Returns:
            GaussianMixture | None: Fitted model or None if fitting failed
        """

        if len(samples) < self.minimum_sample_size:
            warnings.warn(
                f"[Gaussian Mixture Model WARNING] There are not enough {correctness} samples in the \"{class_label}\" class to fit a Gaussian Mixture Model. "
                f"Provided: {len(samples)}. "
                f"Minimum required: {self.minimum_sample_size}. "
                f"We recommend that you use a different safetycage method.",
                UserWarning
        )
        
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
        
        # Try Catch to catch ValueError thrown by sklearn when all fits fail, ignore sklearn warning when some fits fail
        try:
            # If fits fail, sklearn throws a long FitFailedWarning, catch this and let the warning message be provided by the if statement below.
            with warnings.catch_warnings():
                from sklearn.exceptions import FitFailedWarning
                warnings.filterwarnings("ignore", category=FitFailedWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="One or more of the test scores are non-finite:.*",
                    category=UserWarning,
                    module="sklearn.model_selection._search"
                )
                grid_search.fit(samples.reshape(-1, 1))

        except ValueError as e:
            warnings.warn(
                f"[Gaussian Mixture Model WARNING] All GMM fits failed for "
                f"{correctness} samples in class \"{class_label}\". "
                f"This class will be flagged as unreliable.",
                UserWarning
            )
            self.unreliable_classes.add(class_label)
            return None

        # Address the cases where all/some GMM fits fail during CV.
        scores = grid_search.cv_results_["mean_test_score"]

        if np.isnan(scores).any():
            warnings.warn(f"[Gaussian Mixture Model WARNING] There are not enough {correctness} samples in the \"{class_label}\" class to fit a Gaussian Mixture Model.")

            if np.all(np.isnan(scores)):
                # If all scores are NaN .best_estimator_ automatically chooses the first parameter (n_components=1, covariance_type='full').
                warnings.warn(f"All mean BIC scores are NaN values. Hence, model selection is invalid. The results for the \"{class_label}\" class are unreliable and this class will be flagged as unreliable.")
                self.unreliable_classes.add(class_label)
            else:
                warnings.warn(f"{np.sum(np.isnan(scores))} BIC score(s) are NaN values. Model selection for the \"{class_label}\" class may be unreliable.")
        
        return grid_search.best_estimator_


    def _compute_log_pdfs(self, density_correct: GaussianMixture, density_incorrect: GaussianMixture, n_samples: int = int(1e6)) -> dict:
        """
        Compute log-likelihood values for correct and incorrect distributions.

        Args:
            density_correct (GaussianMixture): Model for correct samples
            density_incorrect (GaussianMixture): Model for incorrect samples
            n_samples (int): Number of samples to draw

        Returns:
            dict: Sampled points log-likelihood values
        """

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

        Computes per-layer p-values for each sample and combines them into a global
        p-value using the configured method.
        
        Args:
            x: Input features array
            y: Target values array
            
        Returns:
            np.ndarray: Vector of global combined p-values per sample. Shape depends on s_statistic_source.
        """

        pvalue = self._compute_statistics(x, y)
        
        return self._combine_layer_pvalues(pvalue, len(y), self.test_type_between_layers)


    def _compute_statistics(self, x, y):
        """
        Compute p-values for each sample and layer based on the fitted density estimators and ECDFs.

        Evaluates how likely each sample is under the correct and incorrect distributions.
        Samples belonging to unreliable classes are assigned NaN values.

        Args:
            x: Input data samples
            y: True labels

        Returns:
            numpy.ndarray: Matrix of p-values with shape (num_samples, num_layers)
        """
        selected_layers = self.model_module.selected_layers

        num_samples = len(y)
        num_layers = len(selected_layers)
        
        pvalue = np.full(
            shape = (num_samples, num_layers),
            fill_value = np.inf,
            dtype = np.float64
            )
        
        activations = self.model_module._get_activations(x)

        if self.unreliable_classes:
            warnings.warn(f"The p-value for the classes {self.unreliable_classes} have been set to NaN due to insufficient data to fit the Gaussian Mixture Models. "
                        f"Consider using a different safetycage method.")
        
        for layer_index, layer in enumerate(selected_layers): # for all layers
            for sample_index, y_sample, in enumerate(y): # for all predictions to be tested
                
                # Compute p-values of each sample per layer using ECDF function
                if self.model_module.use_onehot_encoder:
                    class_label = self.classes[np.argmax(y_sample)]
                else:
                    class_label = self.classes[y_sample]
                
                if class_label in self.unreliable_classes:
                    pvalue[sample_index,layer_index] = np.NaN
                    continue

                ## Get the projection vector beta hat and the actication for the sample
                activation = activations[layer][sample_index]
                beta_hat = self.layer_params[layer][class_label]["beta_hat"]
                
                # Compute observed value with respect to beta_hat_i projection for predicted class y[sample_index]:
                activation_projected = np.dot(activation, beta_hat).reshape(1,-1)
                
                # Get the density functions of correctly and incorrectly predicted samples, for the layer
                density_correct = self.layer_params[layer][class_label]["density_correct"]
                density_incorrect = self.layer_params[layer][class_label]["density_incorrect"]
                
                # Compute the s statistic for the sample
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
                    
        return pvalue


    def _combine_layer_pvalues(self, pvalues: np.ndarray, y_len: int, test_type: str | None = None) -> np.ndarray:
        """
        Combine p-values across layers into a global p-value using one of the specified methods:
        - Fisher’s method
        - the Cauchy combination test

        If just one layer of p-values is given, the function simply returns the p-values for that layer.?

        Args:
            pvalues (numpy.ndarray): Per-layer p-values
            y_len (int): Number of samples
            test_type (str): Combination method

        Returns:
            numpy.ndarray: Combined p-values per sample
        """
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
        """
        Flag samples with probability less than or equal (self.s_statistic_source == "correctly") to alpha 
        or probability more than or equal (self.s_statistic_source == "incorrectly") to alpha as incorrect.

        Args:
            statistics (numpy.ndarray): Computed p-values
            alpha (float): Threshold for flagging samples

        Returns:
            numpy.ndarray: Boolean array indicating flagged samples
        """
        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        
        if self.s_statistic_source == "correctly":
            # If alpha argument to flag() function not none, use this and not the one in config-file
            flags = (statistics <= alpha)
            
        # Small p-value indicates sample is correctly classified. Make sure flag = 1 means prediction is deemed to be wrong
        elif self.s_statistic_source == "incorrectly":
            flags = ~(statistics <= alpha)

        return flags

if __name__ == "__main__":
    SPARDACUS(None, None, None)