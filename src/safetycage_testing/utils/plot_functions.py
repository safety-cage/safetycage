# This script contains the library of functions used in the Exaigon safety_cage development
from mailbox import linesep
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from typing import Literal

from ..utils.evaluate import calculate_roc_curve

def annotate_text_box(
    x: float,
    y: float,
    metric_name: str,
    colour: str,
    label_prefix: str = "Best",
    side: str = "auto",          # "left" | "right" | "auto"
    vertical: str = "auto",      # "top" | "bottom" | "auto"
    # layout parameters
    dx: float = 0.0,
    dy: float = 0.0,
    point: bool = True,
    point_size: int = 3,
    zorder: int = 3,
    label_x_offset: float = 0.08,   # horizontal distance from point
    label_y_offset: float = 0.05,   # vertical distance from point
    pad_fraction_x: float = 0.02,   # padding as fraction of axis width
    pad_fraction_y: float = 0.01,   # padding as fraction of axis height
    bbox_pad: float = 0.3,          # value passed to boxstyle round,pad=
    bbox_lw: float = 1.0,
    arrow_lw: float = 1.5,
    ):
    """
    Annotate a point (x,y) with a text box and arrow.
    This is designed specifically for annotating the alpha curve.
    Enforce relative label positions by using side and vertical arguments.
    Hardcode label positions by altering direction factor.

    side:
      - "right": label to the right of point
      - "left":  label to the left of point
      - "auto":  choose based on x location (keeps label inside axes)

    vertical:
      - "top":    label above point
      - "bottom": label below point
      - "auto":   choose based on y location
    
    Improvements to Make:
      - currently, the only function that calls this is plot_alpha_metric_curve, which currently has test metrics labels go upward, and val downward.
      - Fix this.
      - Test for all cases, it's a bit weirdly hardcoded where it works for most cases we'll see, but not all possible cases.
    """

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Choose side/vertical to determine which side of the point that the label is on
    if side == "auto":
        side = "right" if x < (xmin + xmax) / 2 else "left"
    if vertical == "auto":
        vertical = "top" if y < (ymin + ymax) / 2 else "bottom"

    # Signed offsets depending on side/vertical
    sx = +1 if side == "right" else -1
    sy = +1 if vertical == "top" else -1

    # Calculate better position for annotation text to ensure it's inside the figure
    text_x = x + sx * (label_x_offset + dx)
    text_y = y + sy * (label_y_offset + dy)

    pad_x = pad_fraction_x * (xmax - xmin)
    pad_y = pad_fraction_y * (ymax - ymin)
    text_x = min(max(text_x, xmin + pad_x), xmax - pad_x) # 0.02 * (xmax - xmin) <= x <= x + sx * (0.08 + dx)
    text_y = min(max(text_y, ymin + pad_y), ymax - pad_y)
    
    # Align text box in terms relative to the point
    ha = "left" if side == "right" else "right"
    va = "bottom" if vertical == "top" else "top"

    ax.annotate(
        text=f"{label_prefix} alpha: {x:.3f}\nMax {metric_name}: {y:.3f}",
        xy=(x, y),
        xytext=(text_x, text_y),
        bbox=dict(boxstyle=f"round,pad={bbox_pad}", fc="white", ec=colour, lw=bbox_lw),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colour, lw=arrow_lw),
        ha=ha, va=va, zorder=zorder,
    )

    if point:
        ax.plot(x, y, "o", color=colour, markersize=point_size, zorder=zorder)


def plot_alpha_metric_curve(
    alphas: np.ndarray,
    metric_values: np.ndarray,
    thresholds: np.ndarray,
    scores: np.ndarray,
    alpha_opt: float,
    metric_max: float,
    metric_name: str,
    output_path: str,
    alpha_val: float,
    metric_val: float,
    # layout parameters
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save: bool = True,
    opt_label_offset: tuple[float, float] = (0.0, 0.0),
    val_label_offset: tuple[float, float] = (0.0, 0.0),
    zorder: int = -1,
    y_padding: float = 10.0,            # divide (ymax-ymin) by this to get ylim padding, to ensure labels fit in figure
    test_colour: str = "blue",
    val_colour: str = "red",
    test_vertical: str = "top",
    val_vertical: str = "bottom",
    save_dpi: int = 300,
    ) -> None:
    plt.plot(alphas, metric_values, zorder=zorder)
    plt.plot(thresholds, scores, zorder=zorder)

    xlim = (min(min(alphas), min(thresholds)), max(max(alphas), max(thresholds)))
    ymin = min(min(metric_values), min(scores))
    ymax = max(max(metric_values), max(scores))
    ylim = (ymin, ymax + (np.divide(ymax - ymin, y_padding)))

    plt.xlim(*xlim)
    plt.ylim(*ylim)

    dx, dy = opt_label_offset
    annotate_text_box(
        alpha_opt,
        metric_max,
        metric_name,
        label_prefix="Test",
        colour=test_colour,
        vertical=test_vertical,
        dx=dx,
        dy=dy,
    )

    dx, dy = val_label_offset
    annotate_text_box(
        alpha_val,
        metric_val,
        metric_name,
        label_prefix="Val",
        colour=val_colour,
        vertical=val_vertical,
        dx=dx,
        dy=dy,
    )

    plt.xlabel("Alpha")
    plt.ylabel(metric_name)
    if save:
        path = os.path.join(output_path, f"alpha_{metric_name}_curve.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=save_dpi)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    normalize: Literal['true', 'pred', 'all'], 
    output_path: str,
    cmap = plt.cm.Blues,
    save:bool = True,
    ):
    
    # Return confusion matrix for the best threshold
    cm = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize,
    )
    

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Correct prediction", "Incorrect prediction"])
    plt.yticks(tick_marks, ["Correct prediction", "Incorrect prediction"])

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap, ax=plt.gca())

    plt.ylabel("True Misclassification")
    plt.xlabel("Predicted Misclassification")

    if save is True:
        path = os.path.join(output_path, "confusion_matrix.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    plt.close()

def plot_roc_curve(
    tpr, fpr, thresholds,
    output_path: str,
    save:bool = True,
    fig_dimensions = (6,6),
    curve_color: str = "blue",
    fill_in_curve: bool = False,
    fill_in_opacity: float = 0.3,
    diag_color: str = "red",
    diag_linestyle: str = "--",
    xlim: tuple[float, float] = [0.0, 1.0],
    ylim: tuple[float, float] = [0.0, 1.0],
    save_dpi: int = 300
    ):
    
    plt.figure(figsize=fig_dimensions)

    # Plot the ROC curve with correct axes
    plt.plot(fpr, tpr, color=curve_color)

     # [OPTIONALLY: Default = False] Add shaded area under the curve
    if fill_in_curve:
        plt.fill_between(fpr, tpr, alpha=fill_in_opacity, color=curve_color)


    plt.plot(xlim, ylim, color=diag_color, linestyle=diag_linestyle)  # Diagonal line
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
    if save is True:
        path = os.path.join(output_path, "roc_curve.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=save_dpi)
    plt.close()