import sys
sys.path.append('.')

from utils import train_dev_test_split
from sklearn import datasets
from joblib import dump, load


def test_get_bias():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    label = digits.target
    train_fracs = 0.7
    dev_fracs = 0.1
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_fracs, dev_fracs
    )
    model = load("svm_gamma=0.0008_C=2.0.joblib")
    predict = model.predict(x_test)
    checker = predict[0]
    flag = True
    for item in predict:
        if checker != item:
            flag = False
            break;
    assert flag != True
def test_predict_all_classes():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    label = digits.target
    model = load("svm_gamma=0.0008_C=2.0.joblib")
    predict = model.predict(data)
    checker = list(set(label))
    predicted = list(set(predict))
    assert len(checker) == len(predicted)




    
#what more test cases should be there 
#irrespective of the changes to the refactored code.

# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set


# preprocessing gives ouput that is consumable by model

# accuracy check. if acc(model) < threshold, then must not be pushed.

# hardware requirement test cases are difficult to write.
# what is possible: (model size in execution) < max_memory_you_support

# latency: tik; model(input); tok == time passed < threshold
# this is dependent on the execution environment (as close the actual prod/runtime environment)


# model variance? -- 
# bias vs variance in ML ? 
# std([model(train_1), model(train_2), ..., model(train_k)]) < threshold


# Data set we can verify, if it as desired
# dimensionality of the data --

# Verify output size, say if you want output in certain way
# assert len(prediction_y) == len(test_y)

# model persistance?
# train the model -- check perf -- write the model to disk
# is the model loaded from the disk same as what we had written?
# assert acc(loaded_model) == expected_acc 
# assert predictions (loaded_model) == expected_prediction 


















