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

    n_class = len(list(df_train.columns))
    a = 0
    for feature in df_train:
            # Get min and max of feature
        _max = max(df_train[feature].values)
        _min = min(df_train[feature].values)
        quant = []
        diff = _max - _min
            # Dividing in 10 sections
        for i in range(10):
            quant.append(round(_min + ((diff / 10) * (i + 1)), 4))
        _list = []
            # Fit amount of values in sections
        for i in range(9):
            tmp = 0
            for j in df_train[feature].values:
                if j >= quant[i] and j < quant[i + 1]:
                    tmp += 1
            _list.append(tmp)
            # Plot
        plt.subplot(round(int(n_class / 2)), round(int(n_class / 4)), a + 1)
        plt.bar(np.arange(len(_list)), _list)
        plt.xticks(np.arange(len(_list)), quant)
        plt.title(feature)
        a += 1
    plt.show()


if __name__ == '__main__':
    main()
