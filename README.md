# SafetyCage

SafetyCage is a Python package for **detecting misclassified predictions** from machine learning models in classification tasks. It provides a unified interface for multiple statistical detection methods, enabling users to quantify prediction reliability and flag potentially incorrect outputs across different models and datasets easily.

It is available on PyPI here: https://pypi.org/project/safetycage/.

## Background

The idea behind safetycage is that we can find statistics on each predicted sample, and compare that statisitic to some statisitc threshold "alpha" to predict whether the sample prediction was incorrectly classified.

## Description
Machine learning models can produce incorrect predictions with high confidence. SafetyCage addresses this by providing post-hoc misclassification detection methods that operate on model outputs or internal representations.

The package includes several methods:

- MSP (Maximum Softmax Probability)
- DOCTOR (Error probability estimation)
- Mahalanobis (Distance-based statistical testing)
- SPARDACUS (Projection + density estimation approach)

Each method outputs a statistic or p-value that reflects how likely a prediction is to be incorrect.

Alternatively, you can implement your own method by initializing a base class from the safetycage abstract base class, that defines how methods should be implemented.

## Requirements
SafetyCage uses Python 3.11.7. Consider creating an environment for your project with safetycage that uses Python 3.11.7.

## Installation
Install via pip using the command:

```
pip install safetycage
```

The only additional packages you will need to work with safetycage are those required by your dataset and model.

## Visuals
- could put a gif on installing and using the package - don't think this is very necessary, should be obvious

## Tutorials & Examples

A separate repository contains full examples and tutorials at https://github.com/safety-cage/safetycage-tutorials.

## Support
If you encounter issues or have questions:
- Open an issue on the repository
- Check the safetycage-tutorials repo for examples

## Roadmap
- TBD

## Contributing
- unsure

## Authors and acknowledgment
- TBA

## License
MIT License

## Project status
- active and under devlopment