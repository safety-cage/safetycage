---
title: 'Safetycage: A Misclassification Detection Toolkit'
tags:
  - Python
  - Machine learning
  - Deep learning
  - Misclassification Detection
  - AI
authors:
  - give-names: Finn
    dropping-particle: Joel
    surname: Bjervig
    orcid: 0009-0003-8579-1162
    equal-contrib: true
    affiliation: 1
  - given-name: Julia
    surname: Qiu
    equal-contrib: true
    affiliation: 2

affiliations:
 - name: SINTEF AS, Norway
   index: 1
 - name: Waterloo University, Canada
   index: 2

date: 06 March 2026
bibliography: paper.bib
---

# Summary


# Statement of need
`Safetycage` is an package that brings several misclassification detection methods (MDM) under the same framework. The design of safetycage provides a class-based and user-friendly interface with implementations of MDMs such as MSP, and SPARDACUS among others. To provide a fully functioning and general ecosystem for MDM, `Safetycage` relies on model -and datamodules that are wrappers of the model and data choices made by the user. They will need to define some functions such as prediction call syntax, data splits. This way the MDMs are agnostic to model and data, with the caveat that todays implementation currently focuses on classifiers. 

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# State of the field

Several software packages address related aspects of model reliability and data integrity. Libraries such as Cleanlab [@cleanlab] focus primarily on identifying label errors within the training set using confident learning. While effective for dataset curation, it is not focused on detection of misclassifications during inference on novel data. PyTorch-OOD [@pytorch_ood] provides a suite of Out-of-Distribution (OOD) detection methods but is restricted to the PyTorch ecosystem. Safetycage provides similar capabilities but remains framework-agnostic through its data -and model modules, making it easy for users to pass their own classifier and data modalities. Deepchecks [@deepchecks] and Microsoft’s Responsible AI Toolbox [@responsible_ai] offer comprehensive suites for model auditing and integrity checks. However, these are often designed for "at-rest" model evaluation rather than providing a lightweight, extensible API specifically for implementing and testing new misclassification detection algorithms. Packages like NetCal[@netcal] and Scores [@scores] focus on calibrating probability outputs. While calibration is a form of error mitigation, Safetycage focuses on the explicit identification of individual samples likely to be misclassified.

By providing a unified abstraction layer, Safetycage bridges the gap between specialized OOD detection tools and general-purpose model auditing frameworks.

# Software Design

The abstract base classes (ABCs) are important in ensuring users use safetycage in a modular and efficient way. There are 3 ABCs defined:  `SafetyCage`, `ModelModule`, and `DataModule`. Each provides a basis for users to build and create their own classification methods and modules for data and model handling. In this way, users are forced to define the necessary requirements to use safetycage in a meaningful way. This helps avoid users misunderstanding how to use safetycage or misinterpreting the methods. 

Both abstract and default methods are provided to enforce structure while also avoiding repeated code. For example, in the `SafetyCage` ABC, `train_cage()`, `predict()`, and `_compute_statistics()` are all abstract methods. Users are forced to define how they want their `SafetyCage` object to be trained, predict misclassifications, and compute statistics on each sample to determine whether or not that sample was misclassified. Since these depend on the method itself, they must be defined in a derived class.

On the other hand, `flag()`, `find_best_threshold_flag()`, `save_cage()`, and `load_cage()` have a default definition. They are all essential, but generally tend to work in a similar way. Hence, to avoid repeated code across different `safetycage` methods, it is provided in the ABC.

An integral part of using `safetycage` is the `datamodule` and `modelmodule`.

The data module and model module define how the data and model should be handled respectively. The `DataModule` ABC enforces users to define how to load data, transform it, and split into training and validation sets via `_load_data()`, `_transform()`, and `_split()`. The `ModelModule` defines accessor methods to clearly retrieve predictions, preactivations or activations (if any) via `_get_predictions()`, `_get_pre_activations()`, and `_get_activations()`.

A key point of `SafetyCage` is that it accepts both the `DataModule` and `ModelModule` instances. Therefore, different safetycage methods can interact on the same `DataModule` and `ModelModule`, and alternatively, the same safetycage method can interact with different `DataModule` and `ModelModule` instances. This all falls backs to the beginning point of enforcing a certain structure with ABC classes. The modularity and strucutre defined in the use of ABC helps ensure easily compatiability across different datasets, models, and misclassification techniques easily.

It is also worth mentioning that the safetycage methods enforce strong software engineering practices. In development, clarity, simplicity, robust error handling, and comprehensive coverage of edge cases were prioritized. The helper functions provided in `utils` cover multiple cases and do not fail given an incorrect value. An example of this is the `utils.plot_functions.plot_alpha_metric_curve()` function, which allows the user to provide multiple parameters to customize a graph of their statistic threshold values and their corresponding density and metric value. The user can easily define parameters to detail the number of bins in the histogram, the relative location of labels, colours, x-axis and y-axis bounds, image resolution, and more.

Error catching was detailed because, particularly in the case of machine learning, there are often unique cases where assumptions fail. For example, SPARDACUS relies on the assumption that there is a positive amount of correct and incorrect classifications for each class. However, in the case where the model does well, and there are too few, SPARDACUS may not be applicable. The SPARDACUS implementation clearly throws warnings explaining which class has too few samples, when it becomes an issue, and how to address this throughout the misclassification detection pipeline.

In the development of `safetycage`, readability and clarity were of primary focus in naming. Hardcoded values were avoided as much as possible. Documentation was of focus, and each class, method, or function has an associated docstring. However, it is worth mentioning that the docstrings assume the user has a simple machine learning understanding. For example, docstrings do not explain what a training, validation, and test set are. Testing was also performed on different operating systems to ensure that all users can easily use the safetycage package.

# Examples of Use

Here is a simple example of how to use the safetycage method, MSP, the `datamodule`, `modelmodule`.

### Installing

```
pip install safetycage
```

### Data and Model Loading

```python
from safetycage.data import IrisDataModule
from safetycage.model import IrisModelModule

data_module = IrisDataModule(...)
model = joblib.load("model.joblib")
model_module = IrisModelModule(model, ...)
```

### SafetyCage Definition

```python
from safetycage import MSP

safetycage = MSP(model_module=model_module, data_module=data_module)
```

### SafetyCage Training and Prediction

```python
safetycage.train_cage()

statistics = safetycage.predict()

alpha, optimum_alpha_metric = safetycage.find_best_threshold_flag()
```

## 

# Research impact statement

`Gala` has demonstrated significant research impact and grown both its user base
and contributor community since its initial release. The package has evolved
through contributions from over 18 developers beyond the original core developer
(@adrn), with community members adding new features, reporting bugs, and
suggesting new features.

While `Gala` started as a tool primarily to support the core developer's
research, it has expanded organically to support a range of applications across
domains in astrophysics related to Milky Way and galactic dynamics. The package
has been used in over 400 publications (according to Google Scholar) spanning
topics in galactic dynamics such as modeling stellar streams [@Pearson:2017],
Milky Way mass modeling, and interpreting kinematic and stellar population
trends in the Galaxy. `Gala` is integrated within the Astropy ecosystem as an
affiliated package and has built functionality that extends the widely-used
`astropy.units` and `astropy.coordinates` subpackages. `Gala`'s impact extends
beyond citations in research: Because of its focus on usability and user
interface design, `Gala` has also been incorporated into graduate-level galactic
dynamics curricula at multiple institutions.

`Gala` has been downloaded over 100,000 times from PyPI and conda-forge yearly
(or ~2,000 downloads per week) over the past few years, demonstrating a broad
and active user community. Users span career stages from graduate students to
faculty and other established researchers and represent institutions around the
world. This broad adoption and active participation validate `Gala`'s role as
core community infrastructure for galactic dynamics research.

# Mathematics

Let $\mathcal{D}_{\mathrm{train}}=\{(x_i,y_i)\}_{i=1}^{n}$ denote the training set. We first use $\mathcal{D}_{\mathrm{train}}$ to fit the predictive model $f_\theta$. For misclassification detection, methods that require parameter fitting (e.g., fitting score functions, class statistics, or calibration maps) are also trained on training data.

Further, let $\hat y_i=f_\theta(x_i)$ be the model prediction for sample $i$. Define the true misclassification indicator as $m_i=\mathbf{1}\{\hat y_i\neq y_i\}$. Let $s_i=s_\phi(x_i)\in\mathbb{R}$ be the detector score, and define the thresholded detector decision as either $\hat m_i(\alpha)=\mathbf{1}\{s_i\ge\alpha\}$ or $\hat m_i(\alpha)=\mathbf{1}\{s_i\le\alpha\}$. We will define how to find the optimal threshold $\alpha$ later.

Building on the formulation where a predicted misclassification is flagged when a detection score $s_i$ exceeds a threshold $\alpha$, the specific methods implemented in `Safetycage` define their respective scoring functions as follows.

### Maximum Softmax Probability (MSP)
MSP uses the classifier's softmax confidence, where low confidence indicates higher risk of misclassification. In our framework, this is mapped to a score where larger values indicate higher error likelihood, and thresholding produces the detection decision [@msp_placeholder].

### DOCTOR
DOCTOR estimates a rejection rule by contrasting the distributions of correctly and incorrectly classified samples. Samples that are more likely under the misclassified distribution receive higher scores and are flagged when the score exceeds the threshold [@doctor_placeholder].

### Mahalanobis Safetycage
The Mahalanobis method measures how far an input's internal representation is from reference class-conditional activation statistics. Larger distances indicate more atypical behavior and therefore higher misclassification risk [@mahalanobis_placeholder].

### SPARDACUS Safetycage
SPARDACUS learns a projection that emphasizes differences between correctly and incorrectly classified activation distributions, then derives a statistical significance score from that separation. Predictions with stronger evidence of mismatch are assigned higher misclassification scores [@spardacus_placeholder].


### Finding the Optimal Decision Threshold
Given any method-specific score $s_i$, threshold selection is performed on a validation set by treating misclassification detection as a binary decision problem: for each candidate $\alpha$, samples are either flagged ($\hat m_i(\alpha)=1$) or not flagged ($\hat m_i(\alpha)=0$). The optimal threshold is then the one that maximizes a chosen performance metric computed from this induced confusion matrix.

For compact notation, all confusion-matrix terms and derived metrics below are understood to depend on the threshold $\alpha$.

$$
\alpha^{\star}
=
\arg\max_{\alpha\in\mathbb{R}}
J(\alpha),
\quad
J(\alpha) = \mathcal{M}\!\left(\mathrm{TP},\mathrm{FP},\mathrm{TN},\mathrm{FN}\right),
$$

$$ nb
\begin{aligned}
\mathrm{TP} &= \sum_{i=1}^n \mathbf{1}\{\hat m_i(\alpha)=1,\ m_i=1\},\\
\mathrm{FP} &= \sum_{i=1}^n \mathbf{1}\{\hat m_i(\alpha)=1,\ m_i=0\},\\
\mathrm{TN} &= \sum_{i=1}^n \mathbf{1}\{\hat m_i(\alpha)=0,\ m_i=0\},\\
\mathrm{FN} &= \sum_{i=1}^n \mathbf{1}\{\hat m_i(\alpha)=0,\ m_i=1\}.
\end{aligned}
$$

These quantities have the standard interpretation in our setting: $\mathrm{TP}$ counts truly misclassified samples that are correctly flagged, $\mathrm{TN}$ counts correctly classified samples that are correctly not flagged, while $\mathrm{FP}$ and $\mathrm{FN}$ count the two error types of the detector. Thus, optimizing $\alpha$ amounts to selecting the operating point that best balances these outcomes under the selected metric.

Here, $\mathcal{M}(\cdot)$ is an arbitrary scalar metric on the induced confusion matrix. In the SafetyCage papers, the recommended choice is the Matthews Correlation Coefficient (MCC):

$$
\mathcal{M}_{\mathrm{MCC}}=
\frac{\mathrm{TP}\,\mathrm{TN}-\mathrm{FP}\,\mathrm{FN}}
{\sqrt{\big(\mathrm{TP}+\mathrm{FP}\big)
\big(\mathrm{TP}+\mathrm{FN}\big)
\big(\mathrm{TN}+\mathrm{FP}\big)
\big(\mathrm{TN}+\mathrm{FN}\big)}}.
$$

In practice, because $J(\alpha)$ is piecewise constant in $\alpha$, it is sufficient to evaluate candidate thresholds on the sorted unique validation scores (or their midpoints) and select the maximizer. Equivalently, one may sort the validation scores, sweep $\alpha$ across this ordered list, compute $J(\alpha)$ at each step, and return the best threshold.


# Future work
Future work may include regression tasks, where a misclassification is defined as a prediction is a distance $d$ away from the ground truth.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# AI usage disclosure

Generative AI tools were used to create some of the docstrings and documentation of this software. All generated material has been verified by us.

# Acknowledgements
We acknowledge contributions from Pål Johnsen and Filippo Remonato for developing two new misclassification methods, namely Mahalanobis, and SPARDACUS.

EXAIGON THEMIS and SINTEF internal proejcts ?  NFR.

# References