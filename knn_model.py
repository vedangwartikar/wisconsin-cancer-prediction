import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 66)

training_confidence = []
test_confidence = []
neighbors_range = range(1,11)

from sklearn.neighbors import KNeighborsClassifier
for n_neighbors in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train, y_train)
    #Record confidence for train and test sets in array format
    training_confidence.append(knn.score(X_train, y_train))
    test_confidence.append(knn.score(X_test, y_test))
    
print("Training confidence: {}".format(training_confidence) + "\n")
print("Test confidence: {}".format(test_confidence) + "\n")

#Confidence vs Neighbors Visualization
plt.plot(neighbors_range, training_confidence, label = "Training Confidence")
plt.plot(neighbors_range, test_confidence, label = "Test Confidence")
plt.ylabel("Confidence")
plt.xlabel("n_neighbors")
plt.legend()
