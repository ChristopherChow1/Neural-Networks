'''
Author: Christopher Chow
This is where the neural network is implemented
'''

import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import cv2 as cv
import matplotlib.pyplot as plt1
from pathlib import Path

# gets the current working directory
current_working_directory = Path.cwd()
# print(current_working_directory)

data_dir = pathlib.Path('archive').with_suffix('')
"""I want to grab only smile and non-smile"""
image_count = len(list(data_dir.glob('*/*.jpg')))  # use 'smile\*.jpg' to grab smiles
# print(image_count)  # print the total number of images

# create a DataFrame of smiles and label them as 1
smiles = list(data_dir.glob('smile/*'))
non_smiles = list(data_dir.glob('non_smile/*'))
df_smiles = pd.DataFrame(smiles)  # grab a list of smiles
new_df_smiles = df_smiles.assign(smile=1)  # assign 1 as smile
# create a DataFrame of non_smiles and label them as 0
df_non_smiles = pd.DataFrame(non_smiles)
new_df_non_smiles = df_non_smiles.assign(smile=0)

new_df = pd.concat([new_df_smiles, new_df_non_smiles])
new_df.columns = ['image_name', 'smile']
valArray = []
for rows in new_df.index:
    fname = new_df.iat[rows, 0]
    fpath = str(fname)
    fimg = cv.imread(fpath, 0)
    fimgnorm = cv.normalize(fimg, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    # add the resizing function below if necessary: fx and fy changes the size of the image
    # quarter = cv.resize(fimgnorm, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
    npimg = np.asarray(fimgnorm)
    npimg_reshape = npimg.reshape(1, 4096)  # converts to single row
    # npimg_reshape = np.squeeze(npimg)
    valArray.append(npimg_reshape)

new_df['pixel_values'] = valArray
new_df = new_df.drop("image_name", axis=1)  # this gets rid of the name columns

training, testing = train_test_split(new_df, test_size=0.20, shuffle=True)

'''Time to work on Neural Networks'''
"""So I converted a dataframe to np array"""
XTrain = training.loc[:, training.columns != 'smile']

xTrainList = XTrain['pixel_values'].tolist()

yTrain = training.loc[:, 'smile']
yTrainNP = yTrain.to_numpy()

# I HAVE to set y to one hot encoding
one = np.array([1, 0])
zero = np.array([0, 1])
yTrainOneHotList = []
for yIndex in range(len(yTrainNP)):
    if yTrainNP[yIndex] == 0:
        yTrainOneHotList.append(zero)
    else:
        yTrainOneHotList.append(one)
yTrainOneHotArray = np.asarray(yTrainOneHotList)

XTest = testing.loc[:, testing.columns != 'smile']
xTestList = XTest['pixel_values'].tolist()
XTestNP = XTest.to_numpy()
yTest = testing.loc[:, 'smile']
yTestOneHot = []
yTestNP = yTest.to_numpy()
for yTestI in range(len(yTestNP)):
    if yTestNP[yTestI] == 0:
        yTestOneHot.append(zero)
    else:
        yTestOneHot.append(one)
yTestOneHotArray = np.asarray(yTestOneHot)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f_fwd(x, w1, w2):  # feed forward
    # hidden layer
    z1 = x.dot(w1)  # dot product of input and weights
    a1 = sigmoid(z1)  # output of hidden layer

    z2 = a1.dot(w2)  # input to out layer
    a2 = sigmoid(z2)  # output of out layer
    return a2


def init_wt(x, y):
    l = []
    for i in range(x * y):
        l.append(np.random.randn())
    return np.array(l).reshape(x, y)


def loss(out, Y):
    s = (np.square(out - Y))
    s = np.sum(s) / len(Y)
    return s


def backpropagation(x, y, w1, w2, alpha):
    z1 = x.dot(w1)  # input from layer 1
    a1 = sigmoid(z1)  # output from layer 2

    z2 = a1.dot(w2)  # input from layer 2 to layer 3
    a2 = sigmoid(z2)  # output of layer 3

    # error of output layer
    d2 = (a2 - y)
    d1 = np.multiply((w2.dot((d2.T))).T, (np.multiply(a1, 1 - a1)))

    # gradient of w1 and w2
    w1_grad = x.T.dot(d1)
    w2_grad = a1.T.dot(d2)

    # update the weights
    w1 = w1 - (alpha * (w1_grad))
    w2 = w2 - (alpha * (w2_grad))

    return w1, w2  # return the new weights


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
    accuracyList = []
    losses = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_fwd(x[i], w1, w2)
            l.append((loss(out, Y[i])))
            w1, w2 = backpropagation(x[i], Y[i], w1, w2, alpha)
        lossVal = sum(l) / len(x)
        accuracy = (1 - lossVal) * 100
        print("epochs:", j + 1, "======== accuracy:", accuracy)
        accuracyList.append(accuracy)
        losses.append(loss)
    return accuracyList, losses, w1, w2


def predict(x, y, w1, w2):
    print("Predicting on testing data.")
    accuracyList = []
    lossList = []
    l = []
    for i in range(len(x)):
        output = f_fwd(x[i], w1, w2)
        l.append(loss(output, y[i]))
        lossVal = loss(output, y[i])
        accuracy = (1 - loss(output, y[i])) * 100
        print('result:', i + 1, 'predicted:', output, 'actual:', y[i], "accuracy:", accuracy)
        accuracyList.append(accuracy)
        lossList.append(lossVal)
    return accuracyList, lossList


'''
input layer: 4096
hidden layer: 64
output layer: 2
'''
w1 = init_wt(4096, 64)  # set the second parameter to the number of neurons
w2 = init_wt(64, 2)  # also set the first parameter to the same number of neurons

acc, losses, w1, w2 = train(xTrainList, yTrainOneHotArray, w1, w2, 0.01, 100)

finalAcc, finalLoss = predict(xTestList, yTestOneHotArray, w1, w2)
avgAcc = np.sum(finalAcc) / len(finalAcc)  # get the average accuracy for testing data
print("Average Accuracy of testing data:", avgAcc)
plt1.plot(acc)
# plt1.plot(finalAcc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.savefig('accuracy.png')
plt1.show()