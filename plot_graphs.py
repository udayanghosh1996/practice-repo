# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from poplib import CR
from sklearn import datasets, svm, metrics, tree
from itertools import product as pdt

from utils import (
    get_accuracy,
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning_svm,
    #data_viz,
    #pred_image_viz,
    random_split_generator,
    h_param_tuning_dect,
    get_accuracy,
)
from joblib import dump, load


n = int(input("Enter the number of sets of train, dev, test split : "))


train_fracs, dev_fracs, test_fracs = random_split_generator(n)
#print(train_frac,'\n', dev_frac, '\n', test_frac)

# set the hyper parameters SVM
GAMMA = [0.01, 0.005, 0.001, 0.0005, 0.0001]
C = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

# 2. set hyper parameters for decision tree classifier
Criterion = ['gini', 'entropy']
Splitter = ['best', 'random']

# Creating hyperparameters combination for SVM
h_param_comb_svm = pdt(GAMMA,C)
# Creating hyperparameter Combination for Decision Tree Classifier
h_param_comb_dect = pdt(Criterion, Splitter)


# Loading dataset
digits = datasets.load_digits()
#data_viz(digits)
data, label = preprocess_digits(digits)

# Create a svm classifier
clf_svm = svm.SVC()

# create a decision tree classifier
clf_dect = tree.DecisionTreeClassifier()

best_prediction_accuracy_svm =[]
best_prediction_accuracy_dect = []


for i in range(0, n):

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_fracs[i], dev_fracs[i]
    )


    # getting best model for SVM
    best_model_svm, best_metric_svm, best_h_params_svm = h_param_tuning_svm(
        h_param_comb_svm, clf_svm, x_train, y_train, x_dev, y_dev)

    # save the best_model for SVM 
    best_param_config_svm = "_".join([param + "=" + str(best_h_params_svm[param]) for param in best_h_params_svm])
    dump(best_model_svm, "svm_" + best_param_config_svm + ".joblib")

    # getting best model for Decision Tree Classifier
    best_model_dect, best_metric_dect, best_h_params_dect = h_param_tuning_dect(
        h_param_comb_dect, clf_dect, x_train, y_train, x_dev, y_dev)

    # save the best model for Decision Tree Classifier
    best_param_config_dect = "_".join([param + "=" + str(best_h_params_dect[param]) for param in best_h_params_dect])
    dump(best_model_dect, "dect_" + best_param_config_dect + ".joblib")


    # load the best_model for SVM
    best_model_svm = load("svm_" + best_param_config_svm + ".joblib")

    # load the best_model for Decision Tree Classifier
    best_model_dect = load("dect_" + best_param_config_dect + ".joblib")

    # Predict the value of the digit on the test set for SVM Model
    predicted_svm = best_model_svm.predict(x_test)

    # Predict the value of the digit on the test set for Decision Tree Classifier
    predicted_dect = best_model_dect.predict(x_test)

    #pred_image_viz(x_test, predicted_svm)

    # Compute evaluation metrics for SVM
    print(
        f"Classification report for SVM classifier {clf_svm}:\n"
        f"{metrics.classification_report(y_test, predicted_svm)}\n"
    )

    print("Best hyperparameters for SVM Classifier were:")
    print(best_h_params_svm)
    print('\n\n')

    # Compute evaluation metrics for Decision Tree Classifier
    print(
        f"Classification report for Decision Tree classifier {clf_dect}:\n"
        f"{metrics.classification_report(y_test, predicted_dect)}\n"
    )

    print("Best hyperparameters for Decision Tree Classifier were:")
    print(best_h_params_svm)
    print('\n\n\n\n')

    # prediction accuracy for each train, dev, test set for svm and decision tree
    predict_accuracy_svm = get_accuracy(y_test, predicted_svm)
    predict_accuracy_dect = get_accuracy(y_test, predicted_dect)

    # storing accuracies for future use
    best_prediction_accuracy_svm.append(predict_accuracy_svm)
    best_prediction_accuracy_dect.append(predict_accuracy_dect)
print("accuracy list svm: ", best_prediction_accuracy_svm)
print("accuracy list decision tree: ", best_prediction_accuracy_dect)
