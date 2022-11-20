import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score as metric


def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

# other types of preprocessing
# - image : 8x8 : resize 16x16, 32x32, 4x4 : flatteing
# - normalize data: mean normalization: [x - mean(X)]
#                 - min-max normalization
# - smoothing the image: blur on the image




def data_viz(dataset):
    # PART: sanity check visualization of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


# PART: Sanity check of predictions
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


# random split generator
def random_split_generator( seed):
    np.random.seed(seed)
    train = []
    dev = []
    test = []
    train_dev_test = np.array(np.random.random(3))
    train_dev_test /= train_dev_test.sum()
    train= train_dev_test[0]
    dev = train_dev_test[1]
    test = train_dev_test[2]
    return train, dev, test


# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def h_param_tuning_svm(h_param_comb, clf, x_train, y_train, x_dev, y_dev):
    best_accuracy = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for Gamma,c in h_param_comb:

        # PART: setting up hyperparameter
        h_params = {'gamma':Gamma, 'C':c}
        hyper_params = h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # PART: get dev set predictions
        dev_prediction = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        model_accuracy = metric(y_dev, dev_prediction)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model = clf
            best_h_params = h_params
            print("Found new best metric for SVM with :" + str(h_params))
            print("New best val metric for SVM:" + str(model_accuracy))
    return best_model, best_accuracy, best_h_params




def h_param_tuning_dect(h_param_comb, clf, x_train, y_train, x_dev, y_dev):
    best_accuracy = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for Criterion, Splitter in h_param_comb:

        # PART: setting up hyperparameter
        h_params = {'criterion':Criterion, 'splitter':Splitter}
        hyper_params = h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        dev_prediction = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        model_accuracy = metric(y_dev, dev_prediction)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model = clf
            best_h_params = h_params
            print("Found new best metric for Decision Tree Classifier with :" + str(h_params))
            print("New best val metric for Decision Tree Classifier:" + str(model_accuracy))
    return best_model, best_accuracy, best_h_params

def get_accuracy(y_test, predicted):
    accuracy = metric(y_test, predicted)
    return accuracy

def get_accuracy_label_predicted(y_test, predicted):
    correct = sum(y == pred for y, pred in zip(y_test, predicted))
    accuracy = correct/len(y_test)
    return accuracy

def get_mean(arr):
    _mean = np.mean(np.array(arr))
    return _mean
def get_std(arr):
    _std = np.std(np.array(arr))
    return _std