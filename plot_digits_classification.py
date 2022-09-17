"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product as pdt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# model hyperparams
GAMMA = [0.0001,0.0002,0.0003,0.0004,0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
C = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# Create a classifier: a support vector classifier
clf = svm.SVC()


# Split data into 80% train and 10% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True
)

# Split data into 80% train and 10% validation subset
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
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
print("Accuracy and hyperparameter matrix for original image")
print(df)

# considering hyperparameters as best on which we're getting best accuracy on dev dataset
i= dev_accuracy.argmax()

#Printing Best accuracy
print("Best accuracy we can find with parameters for original image\n", "Gamma=",gamma_[i], "C=", c_[i],'\n', "Train Accuracy=%.2f" %(train_accuracy[i]*100), '%', 
"Dev Accuracy=%.2f" %(dev_accuracy[i]*100), '%',"Test Accuracy=%.2f" %(test_accuracy[i]*100), '%' )

print("\n\n\n\n\n\n")

# Image dataset shape and size calculation
images=digits.images
print("Data for Original Image")
print("No of Image present in the dataset is: ", len(images))
print("Shape of entire image dataset is: ", images.shape)
print("Size of each image is: ", images[0].shape)

print("\n\n\n\n\n\n")

# Image resizing to 3 different resolution
image_sizes = [(2, 2), (6, 6), (12, 12)]

#Iteraing over the different image sizes
for image_size in image_sizes:
    image_dataset=[]
    for image in images:
        image_dataset.append(resize(image, image_size, anti_aliasing=False))
    image_dataset=np.array(image_dataset)
    # Split data into 80% train and 10% test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=True)

    # Split data into 80% train and 10% validation subset
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    ##################################################################################################################
    # Creating hyperparameters combinations
    amples=pdt(GAMMA,C)

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
    print("Accuracy and hyperparameter matrix for resized image of size :", image_size)
    print(df)

    # considering hyperparameters as best on which we're getting best accuracy on dev dataset
    j= dev_accuracy.argmax()

    #Printing Best accuracy
    print("Best accuracy we can find with parameters for image size :", image_size, "\n", "Gamma=",gamma_[j], "C=", c_[j],'\n', "Train Accuracy=%.2f" %(train_accuracy[j]*100), '%', 
    "Dev Accuracy=%.2f" %(dev_accuracy[j]*100), '%',"Test Accuracy=%.2f" %(test_accuracy[j]*100), '%' )

    print("\n\n\n\n\n\n")

