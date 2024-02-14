import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# 定义计算熵的函数
from sklearn.tree import DecisionTreeClassifier

import global_settings
from classify_simple_dist import ClassifySimpleDist


def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))


# 定义计算信息增益的函数
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2


def filter_row_list(sorted_id, nn_shapelets, row_list):
    # re_row_list=row_list[sorted_id[:nn_shapelets * 2]]
    re_row_list = []
    count_zero = 0
    count_one = 0

    for i in sorted_id:
        if row_list[i][len(row_list[i]) - 1:] == "0" and count_zero < nn_shapelets:
            count_zero = count_zero + 1
            re_row_list.append(row_list[i])
        elif row_list[i][len(row_list[i]) - 1:] == "1" and count_one < nn_shapelets:
            count_one = count_one + 1
            re_row_list.append(row_list[i])
    print(len(re_row_list))
    return re_row_list


def ex(nn_shapelets):
    df = pd.read_csv('data/l_12_f_6_d_12_ns_40_wsz_3_wsp_1_nc_4_d_32.json_696.csv.multi_sizes.csv')
    X = df
    y = df['label']
    model = DecisionTreeClassifier(criterion='entropy')
    # summarize train and test composition

    row_list = (X.columns)[3:]

    gain_value = [gain(X, row_list[i], 'label') for i in range(len(row_list))]
    X = X[X.columns[3:]]
    # print(gain_value)

    # sort by gain_value and get index
    sorted_id = sorted(range(len(gain_value)), key=lambda k: gain_value[k], reverse=True)
    sorted_list = filter_row_list(sorted_id, nn_shapelets, row_list)
    X = X[sorted_list]
    # print(X)

    scores = cross_val_score(
        model, X, y, cv=10, scoring='accuracy')
    return sum(scores) / len(scores)


def read_data(nn_shapelets):
    df = pd.read_csv('data/l_12_f_6_d_12_ns_40_wsz_3_wsp_1_nc_4_d_32.json_696.csv.multi_sizes.csv')
    X = df
    y = df['label']
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    y_pred = []
    y_true = []
    model = DecisionTreeClassifier(criterion='entropy')
    a = []
    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X.loc[train_ix], X.loc[test_ix]
        train_y, test_y = y.loc[train_ix], y.loc[test_ix]
        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
        test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
        # print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

        row_list = (test_X.columns)[3:]
        gain_value = [gain(X, row_list[i], 'label') for i in range(len(row_list))]
        # print(gain_value)

        # sort by gain_value and get index
        sorted_id = sorted(range(len(gain_value)), key=lambda k: gain_value[k], reverse=True)
        sorted_list = row_list[sorted_id[:nn_shapelets * 2]]
        train_X, test_X = train_X[sorted_list], test_X[sorted_list]

        model.fit(train_X, train_y)
        y_pred_now = model.predict(test_X)  # 测试数据
        y_true.extend(test_y)
        y_pred.extend(y_pred_now)
        c = confusion_matrix(test_y, y_pred_now)
        accuracy = (c[0][0] + c[1][1]) / (c[0][0] + c[1][1] + c[1][0] + c[0][1])
        a.append(accuracy)
    # c = confusion_matrix(y_true,y_pred)
    # # report = classification_report(y_true, y_pred, output_dict=True)
    # accuracy=(c[0][0] + c[1][1]) / (c[0][0] + c[1][1] + c[1][0] + c[0][1])
    return sum(a) / len(a)


# re=[]
# for i in range(100,5,-5):
#     print(i)
#     print("===========")
#     re.append(ex(i))
#     print(re)

def ten_fold_cross_validation(x, y, model):
    # pipeline = make_pipeline(StandardScaler(), model)

    # 设置交叉验证折数cv=10 表示使用带有十折的StratifiedKFold，再把管道和数据集传到交叉验证对象中
    # scores = cross_val_score(pipeline, X=x, y=y, cv=10, n_jobs=1, scoring='accuracy')
    # print('Cross Validation accuracy scores: %s' % scores)
    # print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # 创建一个用于得到不同训练集和测试集样本的索引的StratifiedKFold实例，折数为10
    strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True)
    # 把特征和标签传递给StratifiedKFold实例
    kfold = strtfdKFold.split(x, y)
    y_pred_sum = []
    y_true_sum = []
    # 循环迭代，（K-1）份用于训练，1份用于验证，把每次模型的性能记录下来。
    scores = []
    for k, (train, test) in enumerate(kfold):
        # pipeline.fit(x.iloc[train], y.iloc[train])
        model.fit(x.iloc[train], y.iloc[train])
        # y_pred = pipeline.predict(x.iloc[test])
        y_pred = model.predict(x.iloc[test])
        y_pred_sum.extend(y_pred)
        y_true_sum.extend(y.iloc[test])
    return y_pred_sum, y_true_sum


df = pd.read_csv('data/l_12_f_6_d_12_ns_40_wsz_3_wsp_1_nc_4_d_32.json_696.csv.multi_sizes.csv')
X = df[df.columns[3:]]
y = df['label']
y_pred, y_true = ten_fold_cross_validation(X, y, DecisionTreeClassifier(criterion='entropy'))
c = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)
print((c[0][0] + c[1][1]) / (c[0][0] + c[1][1] + c[1][0] + c[0][1]))
