import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_process
from sys import argv

def Accuracy(theta, x_train, y_train):
    correct = 0
    length = len(x_train)
    prediction = (sigmoid(np.dot(theta, np.transpose(x_train))))
    _y = y_train.values.reshape(-1, 1)
    correct = []
    for i in range(length):
        if prediction[int(_y[i])][i] > 0.6:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = (np.sum(correct) / float(length))*100
    print ('Training accuracy %: {}'.format(accuracy))

def plot_cost(hist_cost, hist_epoch):
    hist_cost = np.transpose(hist_cost)
    plt.plot(hist_epoch, hist_cost[0], label='Ravenclaw')
    plt.plot(hist_epoch, hist_cost[3], label='Hufflepuff')
    plt.plot(hist_epoch, hist_cost[1], label='Slytherin')
    plt.plot(hist_epoch, hist_cost[2], label='Gryffindor')
    plt. xlabel('Epoch')
    plt. ylabel('Cost')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(x, y, theta):
    m = len(x)
    _sum = 0
    for i in range(m):
        _sum += (y[i] * np.log(sigmoid(np.dot(x[i], theta)))) + ((1 - y[i]) * np.log(1 - sigmoid(np.dot(x[i], theta))))
    return -(1 / m) * _sum

def gradient_descent(m, n, x, y, theta, learning_rate, house):
    tmp = np.zeros(n)
    for i in range(m):
        output = 0
        if house == y[i]:
            output = 1
        tmp += (sigmoid(np.dot(x[i], theta)) - output) * x[i]
    theta -= (learning_rate / m) * tmp
    return theta

def training(x, y, theta, learning_rate, nb_epoch, visu=0):
    m = len(y)
    n = len(theta[0])
    hist_cost = []
    hist_epoch = []
    for _iter in range(nb_epoch):
        _cost = []
        for house in range(len(theta)):
            theta[house] = gradient_descent(m, n, x, y, theta[house], learning_rate, house)
            if visu == 1:
                _cost.append(cost(x, y, theta[house]))
        if _iter % 10 == 0 and visu == 1:
            print("cost after epoch {0}: {1}".format(_iter, _cost))
        hist_cost.append(_cost)
        hist_epoch.append(_iter)
    if visu == 1:
        plot_cost(hist_cost, hist_epoch)
    return theta

def save_model(theta):
    print("Writing model to Model.txt")
    fd = open('Model.txt', 'w')
    for i in range(len(theta)):
        for j in range(len(theta[i])):
            fd.write(str(theta[i][j]))
            if j < len(theta[i]) - 1:
                fd.write(', ')
        if i < len(theta) - 1:
            fd.write('\n')
    fd.close()

def main():
        # Get data
    df_train = pd.read_csv("resources/dataset_train.csv")
    df_test = pd.read_csv("resources/dataset_test.csv")

        # Drop unusefull features
    x_train = df_train.drop(['Hogwarts House', 'First Name', 'Last Name', 'Birthday'], axis=1).set_index(['Index'])
    x_test = df_test.drop(['First Name', 'Last Name', 'Hogwarts House', 'Birthday'], axis=1).set_index(['Index'])

    y_train = df_train['Hogwarts House']

    selected_features = {}

        # Get training features
    x_train['Best Hand'] = x_train['Best Hand'].map({'Right': 0, 'Left': 1}).astype(int)
    x_test['Best Hand'] = x_test['Best Hand'].map({'Right': 0, 'Left': 1}).astype(int)
    to_clean, to_fit = data_process.selecte_features(x_train)
    selected_features = {'to_clean': to_clean, 'to_fit': to_fit}
    x_train = data_process.cleaning(x_train, selected_features)

        # Get testing features
    to_clean, to_fit = data_process.selecte_features(x_test)
    selected_features = {'to_clean': to_clean, 'to_fit': to_fit}
    x_test = data_process.cleaning(x_test, selected_features)

    unscal_x_train = x_train
    unscal_x_test = x_test

    selected_features = to_clean + to_fit

        # Features Scaling
    X = x_train.copy()
    col = list(X.columns)
    x_train, x_test = data_process.features_scaling(x_train, x_test, selected_features)
    x_train = x_train.values
    x_test = x_test.values
    outputs = np.unique(y_train)
    y_train = y_train.map({'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}).astype(int)

        # Create Theta
    theta = np.zeros((len(outputs), x_train.shape[1]))
    learning_rate = 0.6
    nb_epoch = 200


    if (len(argv) > 1 and argv[1] == '-v') or (len(argv) > 2 and argv[2] == '-v'):
        theta = training(x_train, y_train, theta, learning_rate, nb_epoch, visu=1)
    else:
        theta = training(x_train, y_train, theta, learning_rate, nb_epoch)

    if (len(argv) > 1 and argv[1] == '-f') or (len(argv) > 2 and argv[2] == '-f'):
        houses_feature_importance = [None] * len(theta)
        features = list(X.columns)
        houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        n_classes = len(houses)
        n_features = len(features) + 1

            # Get houses's features importance
        for i in range(n_classes):
            _sum = np.sum(np.absolute(theta[i]))
            _tmp = []
            for j in theta[i]:
                _tmp.append(abs(j / _sum))
            houses_feature_importance[i] = _tmp

            # Get global features importance
        feature_importance = []
        for i in range(n_features):
            tmp = 0
            for j in range(n_classes):
                tmp += round(houses_feature_importance[j][i], 4)
            feature_importance.append(tmp / n_classes)

            # Drop bias.
        houses_feature_importance = [i[1:] for i in houses_feature_importance]
        feature_importance = feature_importance[1:]

            # Plot features importance
        bar_width = 0.35
        tab = np.arange(len(feature_importance))
        for i in range(len(theta)):
            plt.subplot(2, 2, i + 1)
            plt.bar(tab, feature_importance,width=bar_width, color='g', label="Global features importance")
            plt.bar(tab + bar_width, houses_feature_importance[i], width=bar_width, color='b', label="""{}'s features importance""".format(houses[i]))
            plt.ylim(0, 0.3)
            plt.xticks(tab, features, rotation=90)
            plt.legend()
            plt.title(houses[i])

        feature_importance = pd.DataFrame([i * 100 for i in feature_importance], features, columns=['%']).sort_values(['%'], ascending=False)
        print('Global features importance :')
        print(feature_importance)
        plt.legend()
        plt.show()

    save_model(theta)
    Accuracy(theta, x_train, y_train)

if __name__ == '__main__':
    main()
