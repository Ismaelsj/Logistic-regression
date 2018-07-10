import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
        # Get data
    df_train = pd.read_csv("resources/dataset_train.csv")

        # Keep only numerical values
    to_drop = list(df_train.select_dtypes(['object']).columns)
    to_drop.append('Index')
    to_drop = to_drop[1:]
    df_train = df_train.drop(to_drop, axis=1)
    i = 1

    Gryffindor = df_train[df_train['Hogwarts House'] == 'Gryffindor'].drop(['Hogwarts House'], axis=1)
    Ravenclaw = df_train[df_train['Hogwarts House'] == 'Ravenclaw'].drop(['Hogwarts House'], axis=1)
    Slytherin = df_train[df_train['Hogwarts House'] == 'Slytherin'].drop(['Hogwarts House'], axis=1 )
    Hufflepuff = df_train[df_train['Hogwarts House'] == 'Hufflepuff'].drop(['Hogwarts House'], axis=1)
    df_train = df_train.drop(['Hogwarts House'], axis=1)
    print(Gryffindor.head())
    for feature in df_train:
        plt.subplot(4, 4, i)
        r = df_train[feature].sort_values()
        plt.scatter(np.arange(len(Gryffindor[feature].sort_values())), Gryffindor[feature].sort_values().values, label='{}'.format(feature), color='red', alpha=0.4)
        plt.scatter(np.arange(len(Ravenclaw[feature].sort_values())), Ravenclaw[feature].sort_values().values, label='{}'.format(feature), color='blue', alpha=0.4)
        plt.scatter(np.arange(len(Slytherin[feature].sort_values())), Slytherin[feature].sort_values().values, label='{}'.format(feature), color='green', alpha=0.4)
        plt.scatter(np.arange(len(Hufflepuff[feature].sort_values())), Hufflepuff[feature].sort_values().values, label='{}'.format(feature), color='yellow', alpha=0.4)
        plt.title(feature)
        i += 1
    plt.show()


if __name__ == '__main__':
    main()
