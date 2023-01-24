import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

data = pd.read_csv('penguins.csv')

# pre processing
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])
data['gender'] = data['gender'].fillna((data['gender'].mean()))

# feature Scaling
scaling = MinMaxScaler()
data['bill_length_mm'] = scaling.fit_transform(data[['bill_length_mm']])
data['bill_depth_mm'] = scaling.fit_transform(data[['bill_depth_mm']])
data['flipper_length_mm'] = scaling.fit_transform(data[['flipper_length_mm']])
data['body_mass_g'] = scaling.fit_transform(data[['body_mass_g']])

Y = data.iloc[:, 0]
X = data.iloc[:, 1:6]

class1Features = X.iloc[0:50, :]
class2Features = X.iloc[50:100, :]
class3Features = X.iloc[100:, :]

class1Features = class1Features.sample(frac=1)
class2Features = class2Features.sample(frac=1)
class3Features = class3Features.sample(frac=1)

classes1 = Y.iloc[0:50].replace("Adelie", -1)
classes2 = Y.iloc[50:100].replace("Gentoo", 0)
classes3 = Y.iloc[100:].replace("Chinstrap", 1)

X_train = pd.concat([class1Features.iloc[0:30, :], class2Features.iloc[0:30, :], class3Features.iloc[0:30, :]])
Y_train = pd.concat([classes1.iloc[0:30], classes2.iloc[0:30], classes3.iloc[0:30]])
X_test = pd.concat([class1Features.iloc[30:50, :], class2Features.iloc[30:50, :], class3Features.iloc[30:50, :]])
Y_test = pd.concat([classes1.iloc[30:50], classes2.iloc[30:50], classes3.iloc[30:50]])

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()
Y_train = Y_train.values.tolist()
Y_test = Y_test.values.tolist()

#
c = list(zip(X_train, Y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

c = list(zip(X_test, Y_test))
random.shuffle(c)
X_test, Y_test = zip(*c)




def initializeweights(N_hiddenLayers, N_neurons, bias):
    global weightsList
    weights = np.random.randn() * 0.01
    weightsList = []
    for i in range(N_hiddenLayers + 1):
        if i == 0:
            weights = np.random.rand(N_neurons[i], 5 + bias) * 0.01
        elif i == N_hiddenLayers:
            weights = np.random.rand(3, N_neurons[i - 1]) * 0.01
        else:
            weights = np.random.rand(N_neurons[i], N_neurons[i - 1]) * 0.01
        weightsList.append(weights)




def Model(N_hiddenLayers, N_neurons, eta, m, bias, activationFun):
    outputlayer = []
    initializeweights(N_hiddenLayers, N_neurons, bias)
    for i in range(m):
        for sample in range(len(X_train)):
            features = np.array(X_train[sample])
            features = np.reshape(features, (5, 1))
            if bias:
                features = np.array(X_train[sample])
                features = np.append(features, [1], axis=0)
                features = np.reshape(features, (6, 1))
            F_net_list = forwradStep(N_hiddenLayers, features, weightsList, activationFun)
            if activationFun == "sigmoid":
                derivative_List = backward_step_sigmoid(N_hiddenLayers, F_net_list, sample, weightsList)
            else:
                derivative_List = backward_step_tangent(N_hiddenLayers, F_net_list, sample, weightsList)
            derivative_List.reverse()

            updateweights(derivative_List, N_hiddenLayers, eta, F_net_list, features)

            outputlayer.append(F_net_list[N_hiddenLayers])
            outputlayer[sample] = adjustRes(outputlayer[sample])

        print('train accuracy: ', calcAcc(X_train, outputlayer, Y_train))

        outputlayer = []
    accuracy = Testing(activationFun, N_hiddenLayers, bias)
    return accuracy


def sigmoid(net):
    sigmoidmatrix = np.zeros((len(net), 1))
    for i in range(len(net)):
        f_net = 1 / (1 + np.exp(-net[i]))
        sigmoidmatrix[i] = f_net

    return sigmoidmatrix


def tangentSigmoid(net):
    tangentmatrix = np.zeros((len(net), 1))
    for i in range(len(net)):
        f_net = math.tanh(net[i])  # (1 - np.exp(-net[i])) / (1 + np.exp(-net[i]))
        tangentmatrix[i] = f_net
    return tangentmatrix


def forwradStep(N_hiddenLayers, features, weightsList, activationFun):
    f_net_list = []
    for i in range(0, N_hiddenLayers + 1):
        if i == 0:
            y = np.dot(weightsList[i], features)
        else:
            y = np.dot(weightsList[i], f_net_list[i - 1])
        if (activationFun == "tangent"):
            f_net = tangentSigmoid(y)
        else:
            f_net = sigmoid(y)
        f_net_list.append(f_net)
    return f_net_list


def backward_sigmoid_output_layer(actual, f_net_list, layer_num):
    # for element_wise multiplication
    first_half = np.multiply((actual - f_net_list[layer_num]), (f_net_list[layer_num]))

    derivatives = np.multiply(first_half, (1 - f_net_list[layer_num]))
    return derivatives


def backward_sigmoid_hiddenLayers(derivative, layer_weights, layer_f_nets):
    first_half = np.dot(derivative.T, layer_weights)
    sec_half = np.multiply(layer_f_nets, (1 - layer_f_nets))  # f-prime
    derivative = np.multiply(sec_half, first_half.T)
    return derivative


def backward_step_sigmoid(N_hiddenLayers, f_net_list, sample_indx, weightsList):
    derivatives_list = []
    derivative = backward_sigmoid_output_layer(Y_train[sample_indx], f_net_list, N_hiddenLayers)
    derivatives_list.append(derivative)
    for l in reversed(range(N_hiddenLayers)):
        derivative = backward_sigmoid_hiddenLayers(derivative, weightsList[l + 1], f_net_list[l])
        derivatives_list.append(derivative)
    return derivatives_list


def backward_tangent_output_layer(actual, f_net_list, layer_num):
    first_half = np.multiply((actual - f_net_list[layer_num]), (1 - f_net_list[layer_num]))
    derivatives = np.multiply(first_half, (1 + f_net_list[layer_num]))
    return derivatives


def backward_tangent_hiddenLayers(derivative, layer_weights, layer_f_nets):
    first_half = np.multiply((1 - layer_f_nets), (1 + layer_f_nets))
    second_half = np.dot(derivative.T, layer_weights)
    derivative = np.multiply(first_half, second_half.T)
    return derivative


def backward_step_tangent(N_hiddenLayers, f_net_list, sample_indx, weightsList):
    derivatives_list = []
    derivative = backward_tangent_output_layer(Y_train[sample_indx], f_net_list, N_hiddenLayers)
    derivatives_list.append(derivative)
    for l in reversed(range(N_hiddenLayers)):
        derivative = backward_tangent_hiddenLayers(derivative, weightsList[l + 1], f_net_list[l])
        derivatives_list.append(derivative)
    return derivatives_list


def updateweights(derivatives_list, N_hiddenLayers, eta, F_net_list, features):
    for i in range(N_hiddenLayers + 1):
        features = np.transpose(features)
        if i == 0:
            temp = (np.multiply(derivatives_list[i], features))
            temp = eta * temp
            weightsList[i] = np.add(weightsList[i], temp)
        else:
            temp = np.multiply(derivatives_list[i], (np.transpose(F_net_list[i-1])))
            temp = eta * temp
            weightsList[i] = np.add(weightsList[i], temp)


def Testing(activationFun, N_hiddenLayers, bias):
    global outputLayerRes
    outputLayerRes = []
    for sample in range(len(X_test)):
        features = np.array(X_test[sample])
        features = np.reshape(features, (5, 1))
        if bias:
            features = np.array(X_test[sample])
            features = np.append(features, [1], axis=0)
            features = np.reshape(features, (6, 1))

        F_net_list = forwradStep(N_hiddenLayers, features, weightsList, activationFun)

        outputLayerRes.append(F_net_list[N_hiddenLayers])
        outputLayerRes[sample] = adjustRes(outputLayerRes[sample])

    Accuracy = calcAcc(X_test, outputLayerRes, Y_test)
    print('test accuracy: ', Accuracy)
    return Accuracy


def adjustRes(res):
    maximumNum = max(res)
    for i in range(len(res)):
        if res[i] == maximumNum:
            res[i] = 1
        else:
            res[i] = 0
    return res


def calcAcc(X, outputlayer, y):
    trueCnt = 0
    trueAdelie = 0;
    trueChinstrap = 0;
    trueGentoo = 0;
    AdeAsGen = 0;
    AdeAsChi = 0;
    ChiAsGen = 0;
    ChiAsAde = 0;
    GenAsChi = 0;
    GenAsAde = 0;
    for i in range(0, len(X)):
        if y[i] == -1:
            if outputlayer[i][0] == 1:
                trueCnt = trueCnt + 1
                trueAdelie = trueAdelie + 1
            elif outputlayer[i][1] == 1:
                AdeAsGen = AdeAsGen + 1
            else:
                AdeAsChi = AdeAsChi + 1
        elif y[i] == 0:
            if outputlayer[i][1] == 1:
                trueCnt = trueCnt + 1
                trueGentoo = trueGentoo + 1
            elif outputlayer[i][0] == 1:
                GenAsAde = GenAsAde + 1
            else:
                GenAsChi = GenAsChi + 1
        elif y[i] == 1:
            if outputlayer[i][2] == 1:
                trueCnt = trueCnt + 1
                trueChinstrap = trueChinstrap + 1
            elif outputlayer[i][0] == 1:
                ChiAsAde = ChiAsAde + 1
            else:
                ChiAsGen = ChiAsGen + 1
    print("\t\tClass1\tClass2\tClass3")
    print("Class1 \t", trueAdelie, "\t", AdeAsGen, "\t", AdeAsChi)
    print("Class2 \t", GenAsAde, "\t", trueGentoo, "\t", GenAsChi)
    print("Class3 \t", ChiAsAde, "\t", ChiAsGen, "\t", trueChinstrap)

    accuracy = (trueCnt / len(y)) * 100

    return accuracy


def userTesting(N_hiddenLayers, activationFun, bias, billLen, billDep, flipperLen, gender, bodymass):
    features = [billLen, billDep, flipperLen, gender, bodymass]
    if bias:
        features = np.append(features, [1], axis=0)
        features = np.reshape(features, (6, 1))
    else:
        features = np.reshape(features, (5, 1))
    f_net = forwradStep(N_hiddenLayers, features, weightsList, activationFun)
    res = adjustRes(f_net[N_hiddenLayers])
    if (res[0]):
        ID = "Adelie"
    elif (res[1]):
        ID = "Gentoo"
    else:
        ID = "Chinstrap"
    return ID


