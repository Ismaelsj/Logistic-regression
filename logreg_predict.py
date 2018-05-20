import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_process

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, theta):
    Houses = []
    Index = []
    index = 0
    for pred in x:
        ret = []
        for i in theta:
            z = np.dot(pred, i)
            ret.append(sigmoid(z))
        if ret[0] == max(ret):
            Houses.append('Ravenclaw')
        elif ret[1] == max(ret):
            Houses.append('Slytherin')
        elif ret[2] == max(ret):
            Houses.append('Gryffindor')
        elif ret[3] == max(ret):
            Houses.append('Hufflepuff')
    output = pd.DataFrame({'Hogwarts Houses': Houses})
    print("Writting output to Hogwarts_houses.csv")
    output.to_csv("Hogwarts_houses.csv")

def retrieve_theta():
    fd = open('Model.txt', 'r')
    model = fd.read().split('\n')
    theta = [None] * len(model)
    fd.close()
    for i in range(len(model)):
        theta[i] = [float(j) for j in model[i].split(',')]
    return theta

def main():
        # Get data
    df_train = pd.read_csv("resources/dataset_train.csv")
    df_test = pd.read_csv("resources/dataset_test.csv")

        # Drop unusefull features
    x_train = df_train.drop(['Hogwarts House', 'First Name', 'Last Name', 'Birthday'], axis=1).set_index(['Index'])
    x_test = df_test.drop(['First Name', 'Last Name', 'Hogwarts House', 'Birthday'], axis=1).set_index(['Index'])

    selected_features = {}

        # Get training features
    x_train['Best Hand'] = x_train['Best Hand'].map({'Right': 0, 'Left': 1}).astype(int)
    x_test['Best Hand'] = x_test['Best Hand'].map({'Right': 0, 'Left': 1}).astype(int)

        # Get testing features
    to_clean, to_fit = data_process.selecte_features(x_train)
    selected_features = {'to_clean': to_clean, 'to_fit': to_fit}
    x_train = data_process.cleaning(x_train, selected_features)

    to_clean, to_fit = data_process.selecte_features(x_test)
    selected_features = {'to_clean': to_clean, 'to_fit': to_fit}
    x_test = data_process.cleaning(x_test, selected_features)

    selected_features = to_clean + to_fit

        # Features Scaling
    _, x_test = data_process.features_scaling(x_train, x_test, selected_features)
    x_test = x_test.values

        # Retrieve Thetas
    theta = retrieve_theta()
    predict(x_test, theta)

if __name__ == '__main__':
    main()
