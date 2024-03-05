import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the wine dataset
wine = datasets.load_wine()
X_wine = wine.data
y_wine = wine.target

# Load the iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

# Create kNN classifiers with different values of k
k_values = [1, 3, 9]
knn_classifiers = [KNeighborsClassifier(n_neighbors=k) for k in k_values]

# Create a distance-weighted kNN classifier
knn_weighted = KNeighborsClassifier(weights='distance')

# Function to calculate evaluation metrics for kNN
def evaluate_knn_classifier(classifier, X, y):
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

# Evaluate kNN classifiers with different values of k on the wine dataset
for k, knn_classifier in zip(k_values, knn_classifiers):
    knn_metrics = evaluate_knn_classifier(knn_classifier, X_wine, y_wine)
    print(f"kNN Classifier Metrics on Wine Dataset (k={k}):")
    print(knn_metrics)

# Evaluate distance-weighted kNN on the wine dataset
knn_weighted_metrics = evaluate_knn_classifier(knn_weighted, X_wine, y_wine)
print("\nDistance-Weighted kNN Metrics on Wine Dataset:")
print(knn_weighted_metrics)

# Repeat the evaluation for the iris dataset
for k, knn_classifier in zip(k_values, knn_classifiers):
    knn_metrics = evaluate_knn_classifier(knn_classifier, X_iris, y_iris)
    print(f"\nkNN Classifier Metrics on Iris Dataset (k={k}):")
    print(knn_metrics)

knn_weighted_metrics = evaluate_knn_classifier(knn_weighted, X_iris, y_iris)
print("\nDistance-Weighted kNN Metrics on Iris Dataset:")
print(knn_weighted_metrics)

