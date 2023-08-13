import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import save

def selectKImportance(model, X):

    importantFeatures = model.feature_importances_
    print(importantFeatures)
    Values = np.sort(importantFeatures)[::-1] #SORTED
    print(Values)
    K = importantFeatures.argsort()[::-1][:len(Values[Values>0.00])]
    print(K)
    # save.saveBestK(K)

    # print(' --- begin --- ')
    #
    # for i in K:
    #     print(i, end=', ')
    # print()
    # print(' --- end dumping webserver (425) --- ')
    #
    # C=1
    # for value, eachk in zip(Values, K):
    #     print('rank:{}, value:{}, index:({})'.format(C, value, eachk))
    #
    #      C += 1
    # print('--- end ---')

    ##############################
    # print(Values)
    # print()
    # print(Values[Values>0.00])
    ##############################

    return X[:, K]


def importantFeatures(X, Y):

    model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=500, learning_rate=1.0)
    # 算法:SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重.
    #n_estimators： 整数型，可选参数，默认为50。弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
    # learning_rate： 浮点型，可选参数，默认为1.0。每个弱学习器的权重缩减系数，取值范围为0到1，对于同样的训练集拟合效果，较小的v意味着我们需要更多的弱学习器的迭代次数。

    model.fit(X, Y)

    return selectKImportance(model, X)



