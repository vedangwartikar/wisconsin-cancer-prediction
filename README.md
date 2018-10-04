# wisconsin-cancer-prediction
__Predicting whether the tumor is Begign or Malignent from the Wisconsin Breast Cancer dataset__
- - - -
This project helps in predicting whether the tumor is begign or malignent based on numerous features. Wisconsin Breast Cancer dataset, available under sci-kit learn, provides 569 datapoints along with 30 distinct attributes.

Required dependencies:
* [numpy](http://www.numpy.org/) - Numerical computation
* [sci-kit learn](http://scikit-learn.org/stable/) - Scientific computation
* [mglearn](https://pypi.org/project/mglearn/) - Data visualization

Above dependencies can be installed using pip command in the python shell.

Information about the dataset can be found in dataset_info.py file in the repository.

Refer to the graph confidence-vs-neighbors.png

Algorithm used: K-Nearest Neighbor Classifier
> KNN finds the point in the training set that is closest to the new point. Then it assigns the label of this training point to the new data point. The k in k-nearest neighbors signifies that instead of using only the closest neighbor to the new data point, we can consider any fixed number k of neighbors in the training set
