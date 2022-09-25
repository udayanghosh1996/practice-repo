
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product as pdt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)



# flatten the images
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
Y = digits.target

test_frac=0.1
dev_frac=0.1

def get_train_dev_test_dataset(X, Y, dev_fract, test_fract):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
    return x_train, x_dev, x_test, y_train, y_dev, y_test 

# model hyperparams
GAMMA = [0.0001, 0.0004, 0.0005, 0.0008, 0.001]
C = [0.5, 2.0, 3.0, 4.0, 5.0]
# Create a classifier: a support vector classifier
clf = svm.SVC()

X_train, X_dev, X_test, y_train, y_dev, y_test = get_train_dev_test_dataset(X, Y, dev_frac, test_frac)
gamma_=[]
c_=[]
accuracy=[]
train_accuracy=[]
dev_accuracy=[]
test_accuracy=[]

# Creating hyperparameters combinations
samples=pdt(GAMMA,C)

# iterating over each hyperparameter combinations
for Gamma,c in samples:

#PART: setting up hyperparameter
    hyper_params = {'gamma':Gamma, 'C':c}
    clf.set_params(**hyper_params)


# Learn the digits on the train subset
    clf.fit(X_train, y_train)

# Predict the value of the digit on the train, dev, test datasets
    # Getting training accuracy
    train_prediction = clf.predict(X_train)
    train_accuracy.append(metrics.accuracy_score(y_train, train_prediction))

    # Getting dev accuracy
    dev_prediction = clf.predict(X_dev)
    #print(f"Classification report for classifier {clf}:\n"f"{metrics.classification_report(y_dev, prediction)}\n")
    dev_accuracy.append(metrics.accuracy_score(y_dev,dev_prediction))

    # Getting test accuracy
    test_prediction = clf.predict(X_test)
    test_accuracy.append(metrics.accuracy_score(y_test, test_prediction))
    gamma_.append(Gamma)
    c_.append(c)
gamma_ = np.array(gamma_)
c_ = np.array(c_)
train_accuracy = np.array(train_accuracy)
dev_accuracy = np.array(dev_accuracy)
test_accuracy = np.array(test_accuracy)
combine=np.vstack((gamma_, c_, train_accuracy, dev_accuracy, test_accuracy))
combine=combine.T
column_values = ['GAMMA', 'C', 'Train Accuracy', 'Dev Accuracy', 'Test Accuracy']
df = pd.DataFrame(data = combine, columns=column_values)


# Printing Train, Dev, Test accuracy with each combination of hyperparamameters
print("Accuracy and hyperparameter matrix")
print(df)

# considering hyperparameters as best on which we're getting best accuracy on dev dataset
i= dev_accuracy.argmax()

#Printing Best accuracy
print("Best accuracy we can find with parameters\n", "Gamma=",gamma_[i], "C=", c_[i],'\n', "Train Accuracy=%.2f" %(train_accuracy[i]*100), '%', 
"Dev Accuracy=%.2f" %(dev_accuracy[i]*100), '%',"Test Accuracy=%.2f" %(test_accuracy[i]*100), '%' )

print("\n\n\n\n\n\n")

test_frac_=0.2
dev_frac_=0.2
x_train, x_dev, x_test, y_train, y_dev, y_test = get_train_dev_test_dataset(X, Y, dev_frac_, test_frac_)

for Gamma,c in samples:

#PART: setting up hyperparameter
    hyper_params = {'gamma':Gamma, 'C':c}
    clf.set_params(**hyper_params)


# Learn the digits on the train subset
    clf.fit(x_train, y_train)

# Predict the value of the digit on the train, dev, test datasets
    # Getting training accuracy
    train_prediction = clf.predict(x_train)
    train_accuracy.append(metrics.accuracy_score(y_train, train_prediction))

    # Getting dev accuracy
    dev_prediction = clf.predict(x_dev)
    #print(f"Classification report for classifier {clf}:\n"f"{metrics.classification_report(y_dev, prediction)}\n")
    dev_accuracy.append(metrics.accuracy_score(y_dev,dev_prediction))

    # Getting test accuracy
    test_prediction = clf.predict(x_test)
    test_accuracy.append(metrics.accuracy_score(y_test, test_prediction))
    gamma_.append(Gamma)
    c_.append(c)
gamma_ = np.array(gamma_)
c_ = np.array(c_)
train_accuracy = np.array(train_accuracy)
dev_accuracy = np.array(dev_accuracy)
test_accuracy = np.array(test_accuracy)
combine=np.vstack((gamma_, c_, train_accuracy, dev_accuracy, test_accuracy))
combine=combine.T
column_values = ['GAMMA', 'C', 'Train Accuracy', 'Dev Accuracy', 'Test Accuracy']
df = pd.DataFrame(data = combine, columns=column_values)


# Printing Train, Dev, Test accuracy with each combination of hyperparamameters
print("Accuracy and hyperparameter matrix with different train dev test split")
print(df)

# considering hyperparameters as best on which we're getting best accuracy on dev dataset
i= dev_accuracy.argmax()

#Printing Best accuracy
print("Best accuracy we can find with parameters with different train dev test split\n", "Gamma=",gamma_[i], "C=", c_[i],'\n', "Train Accuracy=%.2f" %(train_accuracy[i]*100), '%', 
"Dev Accuracy=%.2f" %(dev_accuracy[i]*100), '%',"Test Accuracy=%.2f" %(test_accuracy[i]*100), '%' )

print("\n\n\n\n\n\n")