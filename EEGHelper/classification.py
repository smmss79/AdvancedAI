"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.

"""

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  VotingClassifier

names = [
    "Logistic Regression",
    # "Nearest Neighbors",
    # "Linear SVM",
    "RBF SVM",
    "RBF SVM1",
    "RBF SVM2",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
]

classifiers = [
    LogisticRegression(max_iter=1000000,multi_class='multinomial', random_state=1),
    # KNeighborsClassifier(5),
    # SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    OneVsRestClassifier(SVC(gamma=2, C=1)),
    OneVsOneClassifier(SVC(gamma=2, C=1)),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(hidden_layer_sizes=(50,4),max_iter=100000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]


def compare_classifiers(X_train,X_test,y_train,y_test):
    # iterate over classifiers
    # i=251

    score = []
    final_models = []
    for name, clf in zip(names, classifiers):
        print("\nmodel: ",name)

        clf = make_pipeline(StandardScaler(), clf)

        a= time.time_ns()
        clf.fit(X_train, y_train)
        b= time.time_ns()
        print("Train Time: ",b-a)

        score.append( clf.score(X_test, y_test) )
        final_models.append(clf)     
        print("Score: ",score[-1])



    # plt.tight_layout()
    plt.scatter(names, score)
    # plt.show()

    return final_models

def voting_classification(X_train,X_test,y_train,y_test):

    print("\nVoting: ")


    eclf1 = VotingClassifier(estimators=list( zip(names,classifiers) ), voting='hard')
    a= time.time_ns()
    eclf1 = eclf1.fit(X_train, y_train)
    b= time.time_ns()

    print("Train Time: ",b-a)
    print("Score: ",eclf1.score(X_test,y_test))

