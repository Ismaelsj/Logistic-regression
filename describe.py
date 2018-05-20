import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("./resources/dataset_train.csv")
    df = df.drop(['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', ], axis=1)
    features = []
    for feature in df:
        features.append(feature)

    index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    descript = pd.DataFrame(index, columns=['Describe'])
    descript = descript.set_index(['Describe'])

    for feature in features:
        Value = []
        Value.append(len(df[feature]))
        Value.append(df[feature].mean())
        Value.append(df[feature].std())
        Value.append(df[feature].min())
        Value.append(df[feature].quantile(0.25))
        Value.append(df[feature].quantile(0.5))
        Value.append(df[feature].quantile(0.75))
        Value.append(df[feature].max())
        descript[feature] = Value
    print(descript)


if __name__ == '__main__':
    main()
