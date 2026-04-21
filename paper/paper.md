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

# Software design

modules `safetycage.datamodule` and `safetycage.modelmodule`

- ABC enforce users to have a certain setup
- Hence, even if we switch to use different datasets (hence different data modules) and models (hence different model modules), the code shouldn't have to change (much)

- Allow results to update with given cases (ex. plot_alpha)

- Implement error catching in edge cases (ex. SPARDACUS when too few incorrect predictions)

- General good SWE practices (Documentation
Naming Clarity
Avoid repeated code
Avoid hardcoded values
)

- Assumptions we make about the user (simple machine learning understanding?, we don't explain what a train, val, test set is ...)

- Testing with different OS

`Gala`'s design philosophy is based on three core principles: (1) to provide a
user-friendly, modular, object-oriented API, (2) to use community tools and
standards (e.g., Astropy for coordinates and units handling), and (3) to use
low-level code (C/C++/Cython) for performance while keeping the user interface
in Python. Within each of the main subpackages in `gala` (`gala.potential`,
`gala.dynamics`, `gala.integrate`, etc.), we try to maintain a consistent API
for classes and functions. For example, all potential classes share a common
base class and implement methods for computing the potential, forces, density,
and other derived quantities at given positions. This also works for
compositions of potentials (i.e., multi-component potential models), which
share the potential base class but also act as a dictionary-like container for
different potential components. As another example, all integrators implement a
common interface for numerically integrating orbits. The integrators and core
potential functions are all implemented in C without support for units, but the
Python layer handles unit conversions and prepares data to dispatch to the C
layer appropriately.Within the coordinates subpackage, we extend Astropy's
coordinate classes to add more specialized coordinate frames and
transformations that are relevant for Galactic dynamics and Milky Way research.

# Explaining Pivotal Functions & Methods

- train_cage()
- predict()
- find_best_threshold()
- flag()
- find_best_threshold_flag()

```python
train_cage()

3+1

a = 1

b = 2

a + b

```

# Examples of Use

## Installing

## Data Loading

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

$$
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