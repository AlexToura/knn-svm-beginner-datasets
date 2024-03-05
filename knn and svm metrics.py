import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the wine dataset
wine = datasets.load_wine()
X_wine = wine.data
y_wine = wine.target
# Load the iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
# Create kNN and SVM classifiers
knn_classifier = KNeighborsClassifier()
svm_classifier = SVC()
# Function to calculate evaluation metrics
def evaluate_classifier(classifier, X, y):
 kf = KFold(n_splits=10, shuffle=True, random_state=42)
 accuracy_scores = cross_val_score(classifier, X, y, cv=kf,
scoring='accuracy')
 precision_scores = cross_val_score(classifier, X, y, cv=kf,
scoring='precision_macro')
 recall_scores = cross_val_score(classifier, X, y, cv=kf, scoring='recall_macro')
 f1_scores = cross_val_score(classifier, X, y, cv=kf, scoring='f1_macro')
 return {
 'Accuracy': np.mean(accuracy_scores),
 'Precision': np.mean(precision_scores),
 'Recall': np.mean(recall_scores),
 'F-Measure': np.mean(f1_scores)
 }
# Evaluate kNN on the wine dataset
knn_wine_metrics = evaluate_classifier(knn_classifier, X_wine,
y_wine)
print("kNN Classifier Metrics on Wine Dataset:")
print(knn_wine_metrics)
# Evaluate SVM on the wine dataset
svm_wine_metrics = evaluate_classifier(svm_classifier, X_wine,
y_wine)
print("\nSVM Classifier Metrics on Wine Dataset:")
print(svm_wine_metrics)
# Evaluate kNN on the iris dataset
knn_iris_metrics = evaluate_classifier(knn_classifier, X_iris,
y_iris)
print("\nkNN Classifier Metrics on Iris Dataset:")
print(knn_iris_metrics)
# Evaluate SVM on the iris dataset
svm_iris_metrics = evaluate_classifier(svm_classifier, X_iris,
y_iris)
print("\nSVM Classifier Metrics on Iris Dataset:")
print(svm_iris_metrics)
