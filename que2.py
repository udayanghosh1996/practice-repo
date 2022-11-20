from poplib import CR
from sklearn import datasets, svm, metrics, tree
from itertools import product as pdt
import numpy as np
from joblib import dump, load
import argparse
import os

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
    get_mean,
    get_std,
)



parser = argparse.ArgumentParser()
parser.add_argument("--clf_name",dest='clf_name', type=str)

        
parser.add_argument("--random_state",dest='random_state', type=int)

args = parser.parse_args()


seed = args.random_state
clf = args.clf_name


# set the hyper parameters SVM
GAMMA = [0.0001, 0.0004, 0.0005, 0.0008, 0.001]
C = [0.5, 2.0, 3.0, 4.0, 5.0]

# 2. set hyper parameters for decision tree classifier
Criterion = ['gini', 'entropy']
Splitter = ['best', 'random']

# Creating hyperparameters combination for SVM
h_param_comb_svm = pdt(GAMMA,C)
# Creating hyperparameter Combination for Decision Tree Classifier
h_param_comb_dect = pdt(Criterion, Splitter)


#
root = os.getcwd()
model_path = os.path.join(root,'models')
print(model_path)
result_path = os.path.join(root, 'results')

if clf == 'svm':
    model = svm.SVC()
elif clf == 'tree':
    model = tree.DecisionTreeClassifier()
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
label = digits.target

train_fracs, dev_fracs, test_fracs = random_split_generator(seed)
x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_fracs, dev_fracs
    )
if clf == 'svm':
    # getting best model for SVM
    best_model_svm, best_metric_svm, best_h_params_svm = h_param_tuning_svm(
        h_param_comb_svm, model, x_train, y_train, x_dev, y_dev)
        # save the best_model for SVM 
    best_param_config_svm = "_".join([param + "=" + str(best_h_params_svm[param]) for param in best_h_params_svm])
    file_name = os.path.join(model_path,"svm_" + best_param_config_svm + ".joblib")
    dump(best_model_svm, file_name)

elif clf == 'tree':

    # getting best model for Decision Tree Classifier
    best_model_dect, best_metric_dect, best_h_params_dect = h_param_tuning_dect(
        h_param_comb_dect, model, x_train, y_train, x_dev, y_dev)

    # save the best model for Decision Tree Classifier
    best_param_config_dect = "_".join([param + "=" + str(best_h_params_dect[param]) for param in best_h_params_dect])
    file_name = os.path.join(model_path,"dect_" + best_param_config_dect + ".joblib")
    dump(best_model_dect, file_name)

if clf == 'svm':
    model = load(os.path.join(model_path,"svm_" + best_param_config_svm + ".joblib"))

elif clf == 'tree':
    model = load(os.path.join(model_path,"dect_" + best_param_config_dect + ".joblib"))

predicted = model.predict(x_test)

print(
        f"Classification report for classifier :\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
result_file = os.path.join(result_path, clf + '_' + str(seed) + '.txt')
with open(result_file, 'w') as file:
    file.write("test accuracy:")
    file.write(str(metrics.accuracy_score(y_test, predicted)))
    file.write('\ntest macro-f1:')
    file.write(str(metrics.f1_score(y_test, predicted, average='macro')))
