# Class-based implementation of modified logistic regression for PU learning

This repository is a fork of the code provided by the authors of [A Modified Logistic Regression for Positive and Unlabeled Learning
](https://ieeexplore.ieee.org/document/9048765).

This repository
* Refactors the original implementation such that MLR is defined through a class, with standard `.fit()` and `.predict()` methods.
* Refactors the project into a Python package, such that it can be managed as a dependency in `requirements.txt` in other projects

All possible attempts have been made to ensure that the working of MLR here is the same as in the original paper and repository, but please use this code at your own risk. I am not affiliated with the authors of the paper in any way.
