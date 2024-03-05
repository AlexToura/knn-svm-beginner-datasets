import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, KFold
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

# Create SVM classifiers with different kernel functions
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf')
svm_poly = SVC(kernel='poly')

# Function to calculate evaluation metrics for SVM
def evaluate_svm_classifier(classifier, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(classifier, X, y, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(classifier, X, y, cv=kf, scoring='precision_macro')
    recall_scores = cross_val_score(classifier, X, y, cv=kf, scoring='recall_macro')
    f1_scores = cross_val_score(classifier, X, y, cv=kf, scoring='f1_macro')
    return {
        'Accuracy': np.mean(accuracy_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'F-Measure': np.mean(f1_scores)
    }

# Evaluate SVM classifiers with different kernel functions on the wine dataset
for classifier, kernel in zip([svm_linear, svm_rbf, svm_poly], ['Linear', 'RBF', 'Polynomial']):
    svm_metrics = evaluate_svm_classifier(classifier, X_wine, y_wine)
    print(f"SVM Classifier Metrics on Wine Dataset (Kernel={kernel}):")
    print(svm_metrics)

# Evaluate SVM classifiers with different kernel functions on the iris dataset
for classifier, kernel in zip([svm_linear, svm_rbf, svm_poly], ['Linear', 'RBF', 'Polynomial']):
    svm_metrics = evaluate_svm_classifier(classifier, X_iris, y_iris)
    print(f"\nSVM Classifier Metrics on Iris Dataset (Kernel={kernel}):")
    print(svm_metrics)
