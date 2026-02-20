import numpy as np
from functools import partial
from typing import List, Dict, Tuple, Union, Any, Optional
from sklearn import metrics

# from ..model.components.safetycage import SafetyCage
from ..ABC.safetycage import SafetyCage

def precision(TP, TN, FP, FN):
    denom = TP + FP
    return np.divide(TP, denom, out=np.zeros_like(TP, dtype=float), where=denom > 0)

def recall(TP, TN, FP, FN):
    denom = TP + FN
    return np.divide(TP, denom, out=np.zeros_like(TP, dtype=float), where=denom > 0)

def specificity(TP, TN, FP, FN):
    denom = TN + FP
    return np.divide(TN, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)

def NPV(TP, TN, FP, FN):
    denom = TN + FN
    return np.divide(TN, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)

def MCC(TP, TN, FP, FN):
    """
    Vectorized version of MCC calculation.
    TP, TN, FP, FN are numpy arrays of the same length.
    """

    # 2. Calculate numerator and denominator arrays
    numerator = (TP * TN) - (FP * FN)
    # The four terms of the denominator
    d1, d2, d3, d4 = (TP + FP), (TP + FN), (TN + FP), (TN + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_denom = 0.5 * (np.log(d1) + np.log(d2) + np.log(d3) + np.log(d4))
        denom = np.exp(log_denom) # Initialize output array mcc = np.zeros_like(numerator)

    # Ensure output is float to avoid UFuncTypeError
    out_arr = np.zeros_like(numerator, dtype=float)
    mcc = np.divide(numerator, denom, out=out_arr, where=denom != 0)

    return mcc

metric_functions = {
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "NPV": NPV,
    "MCC": MCC,
}

def calculate_auroc(safetycage:SafetyCage, y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate the Area Under the ROC Curve (AUROC) manually using the safety cage.
    
    Args:
        y_true (np.ndarray): True binary labels (incorrect predictions).
        y_scores (np.ndarray): Statistics/scores from the classifier.
        
    Returns:
        float: The computed AUROC value.
    """
    
    # Sort unique thresholds in descending order to compute ROC points
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    # Add infinity as the first threshold to ensure we start at (0,0)
    thresholds = np.append(thresholds, np.inf)
    
    # Initialize arrays to store TPR and FPR values
    tpr_values = []
    fpr_values = []
    
    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        # Get flags using the safety cage
        flags = safetycage.flag(y_scores, threshold)
        
        # Calculate confusion matrix components
        confusion_rates = calculate_confusion_rates(y=y_true, y_pred=flags)
        
        # Calculate TPR (recall) and FPR (1 - specificity)
        tpr = recall(**confusion_rates)
        fpr = 1.0 - specificity(**confusion_rates)
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Convert to numpy arrays
    tpr_values = np.array(tpr_values)
    fpr_values = np.array(fpr_values)
    
    # Calculate AUC using the trapezoidal rule
    # Sort by FPR to ensure correct calculation
    sorted_indices = np.argsort(fpr_values)
    fpr_sorted = fpr_values[sorted_indices]
    tpr_sorted = tpr_values[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc_value = np.trapz(y=tpr_sorted, x=fpr_sorted)
    
    return float(auc_value)


def calculate_confusion_rates(y:np.ndarray,y_pred:np.ndarray):
    """
    For misclassification y is incorrect predictions,
    and predicted misclassifications is y_pred

    Args:
        y (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels

    Returns:
        dict: true and false rates for positive and negative predictions
    """
    return {
        "TP": np.sum((y == 1) & (y_pred == 1)).item(),
        "TN": np.sum((y == 0) & (y_pred == 0)).item(),
        "FP": np.sum((y == 0) & (y_pred == 1)).item(),
        "FN": np.sum((y == 1) & (y_pred == 0)).item()
    }


def calculate_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    metric_functions: Dict[str, callable] = metric_functions
):
    """
    For misclassification y is incorrect predictions,
    and predicted misclassifications is y_pred
    Args:
        y (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
    """
    
    confusion_rates = calculate_confusion_rates(
        y=y,
        y_pred=y_pred,
    )
    
    metrics_dict = {}
    for name, func in metric_functions.items():
    # Force the result to a standard Python float
        metrics_dict[name] = float(func(**confusion_rates))
        
    return metrics_dict

"""
for t in thres:


"""

def find_best_threshold(y_true, y_probs, metric_fn, greater_is_better=True, leq=True):
    """
    Find thresholds that maximize a given metric.

    Args:
        greater_is_better (bool): the greater the metric value, the better it is
        leq (bool) i.e. "less than or equal to":  Whether we are looking for smaller (ex. MSP) or larger (ex. DOCTOR) probabilities.
    """
    # 1. Sort probabilities in ASCENDING order (because lower = more likely to flag)
    asc_indices = np.argsort(y_probs)

    if not leq:
        asc_indices = asc_indices[::-1]
        
    y_probs = y_probs[asc_indices]
    y_true = y_true[asc_indices]

    # 2. Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_probs))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # 3. Calculate TP and FP
    # Since we flag if statistics <= alpha, as the threshold (alpha) increases, 
    # we include MORE samples in our 'flagged' (Positive) set.
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    
    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos
    
    fns = total_pos - tps
    tns = total_neg - fps
    
    # 4. Calculate scores
    scores = metric_fn(TP=tps, TN=tns, FP=fps, FN=fns)

    # 5. Find the best
    best_idx = np.argmax(scores) if greater_is_better else np.argmin(scores)
    
    return {
        "alphas": y_probs[threshold_idxs], 
        "metric_values": scores,
        "alpha_opt": float(y_probs[threshold_idxs[best_idx]]),
        "metric_max": float(scores[best_idx]),
        "metric_name": metric_fn.__name__
    }

def calculate_negative_metric(alpha:float, metric_fn, statistics:np.ndarray, safecage:SafetyCage, incorrect_predictions:np.ndarray):
    """
    Calculates the negative metric value for use in optimization to find the optimal threshold alpha.
    The function is negated because optimization algorithms typically minimize rather than maximize.

    Args:
        alpha (float): Threshold parameter used for the statistics
        metric_fn: Function to calculate the metric to optimize
        statistics (np.ndarray): Statistics of whether a prediction was correct or incorrect
        safecage (SafetyCage): SafetyCage object
        incorrect_predictions (np.ndarray): Ground truth of whether a prediction was correct or incorrect

    Returns:
        float: Negative metric value for optimization
    """
    try:
        # Get predictions from safety cage
        flags = safecage.flag(statistics, alpha)
        
        # Calculate confusion rates
        confusion_rates = calculate_confusion_rates(
            y=incorrect_predictions,
            y_pred=flags
        )
        
        # Calculate the metric
        metric_value = metric_fn(**confusion_rates)

        # Return negative value for minimization
        return -metric_value

    except Exception as e:
        # If there's an error, return a very large value to discourage this alpha
        print(f"Error calculating metric at alpha={alpha}: {str(e)}")
        return 1e6  # A large value to ensure this alpha is not chosen




def calculate_roc_curve(safetycage: SafetyCage, y_true: np.ndarray, statistics: np.ndarray, num_thresholds: int = 1e3) -> tuple:
    """
    Calculate the ROC curve data points using the SafetyCage's own flag function.
    This handles different flag implementations across various SafetyCage implementations.
    
    Args:
        safetycage (SafetyCage): The SafetyCage instance to use for flagging
        y_true (np.ndarray): True binary labels (incorrect predictions)
        statistics (np.ndarray): Statistics/scores computed from the SafetyCage
        num_thresholds (int, optional): Number of threshold points to use. Defaults to 100.
        
    Returns:
        tuple: A tuple containing (fpr, tpr, thresholds)
            - fpr (np.ndarray): False positive rates
            - tpr (np.ndarray): True positive rates
            - thresholds (np.ndarray): Threshold values used
    """
    
    thresholds = np.linspace(0, 1, int(num_thresholds))

    # Initialize arrays to store TPR and FPR values
    tpr_values = []
    fpr_values = []
    
    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        # Get flags using the safety cage's own flag function
        flags = safetycage.flag(statistics, threshold)
        
        # Calculate confusion matrix components
        confusion_rates = calculate_confusion_rates(y=y_true, y_pred=flags)
        
        # Calculate TPR (recall) and FPR (1 - specificity)
        tpr = recall(**confusion_rates)
        fpr = 1.0 - specificity(**confusion_rates)
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Convert to numpy arrays
    tpr_values = np.array(tpr_values)
    fpr_values = np.array(fpr_values)
    
    return {
        "fpr": fpr_values,
        "tpr": tpr_values,
        "thresholds": thresholds
    }

