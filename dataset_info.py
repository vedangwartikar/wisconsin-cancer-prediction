import numpy as np

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print("Keys of cancer dataset: {}".format(cancer.keys()) + "\n")

print("Shape of cancer data: {}".format(cancer.data.shape) + "\n")

print("Sample counts per class: {}".format({n:v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))