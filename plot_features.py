import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y_max_lim(global_features_importance, class_features_importance):
    _max = []
    _max.append(np.amax(np.amax(class_features_importance)))
    _max.append(np.amax(global_features_importance))
    _max = max(_max)
    return (_max + _max / 3)

def features_importance(features_name, class_name, theta, bias=True):
    class_features_importance = [None] * len(theta)
    n_class = len(class_name)
    n_features = len(features_name) + 1
    if bias == False:
        n_features = len(features_name)

        # Get houses's features importance
    for i in range(n_class):
        _sum = np.sum(np.absolute(theta[i]))
        _tmp = []
        for j in theta[i]:
            _tmp.append(abs(j / _sum))
        class_features_importance[i] = _tmp

        # Get global features importance
    global_features_importance = []
    for i in range(n_features):
        tmp = 0
        for j in range(n_class):
            tmp += round(class_features_importance[j][i], 4)
        global_features_importance.append(tmp / n_class)

        # Drop bias
    if bias == True:
        class_features_importance = [i[1:] for i in class_features_importance]
        global_features_importance = global_features_importance[1:]

        # Plot features importance
    bar_width = 0.35
    tab = np.arange(len(global_features_importance))
    _ylim = y_max_lim(global_features_importance, class_features_importance)
    for i in range(len(theta)):
        plt.subplot(round(int(n_class / 2)), round(int(n_class / 2)), i + 1)
        plt.bar(tab, global_features_importance, width=bar_width, color='g', label="Global features importance")
        plt.bar(tab + bar_width, class_features_importance[i], width=bar_width, color='b', label="""{}'s features importance""".format(class_name[i]))
        plt.ylim(0, _ylim)
        plt.xticks(tab, features_name, rotation=90)
        plt.legend()
        plt.title(class_name[i])

    class_dict = {}
    for i in range(n_class):
        class_dict[class_name[i]] = [round(j * 100, 4) for j in class_features_importance[i]]
    global_features_importance = pd.DataFrame([i * 100 for i in global_features_importance], features_name, columns=['Global features']).sort_values(['Global features'], ascending=False)
    class_features_importance = pd.DataFrame(class_dict, features_name).reindex(list(global_features_importance.index))
    print('Global features importance :')
    print(global_features_importance)
    print("")
    print('Class features importance :')
    print(class_features_importance)
    print("")
    plt.legend()
    plt.show()

