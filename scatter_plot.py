import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
        # Get data
    df_train = pd.read_csv("resources/dataset_train.csv")

        # Keep only numerical values
    to_drop = list(df_train.select_dtypes(['object']).columns)
    to_drop.append('Index')
    df_train = df_train.drop(to_drop, axis=1)
    i = 1
    for feature in df_train:
        plt.subplot(4, 4, i)
        r = df_train[feature].sort_values()
        plt.scatter(np.arange(len(r)), r)
        plt.legend()
        i += 1
    plt.show()


if __name__ == '__main__':
    main()
