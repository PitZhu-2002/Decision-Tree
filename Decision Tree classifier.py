import math
from copy import copy
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


class DT:
    def __init__(self, type = 'entropy', container = []):
        self.type = type
        self.container = container
        self.signal = 0
        self.y_set = None

    def entropy(self, y):
        e = 0
        unique_elements, counts = np.unique(y, return_counts=True)
        count_dict = dict(zip(unique_elements, counts))
        for value in count_dict.values():
            p = value / len(y)
            e = e - p * math.log2(p)
        return e

    def compute(self,y , H):    # 计算划分标准如 信息增益
        H_c = copy(H)
        if self.type == 'entropy':
            size = sum([len(part) for part in y])
            for feature in y:
                H_c = H_c - self.entropy(feature) * len(feature) / size
            return H_c
        # elif self.type == 'gini':
        #     pass
        #     return H_c

    def assign_value(self, cn, rt):
        layer, k_bf, k_now, fv = cn[0], cn[1], cn[2], cn[3]
        self.container[layer - 1][(k_bf, k_now)][fv] = (k_now,rt)
        return str(k_bf) + str(cn[2]) + str(cn[3]) # 返回之前划分的 feature 方便接下来保存, 用字符串空间小

    def stop_criterion(self, rest_y, rest_feature, cn):
        if len(rest_feature) == 0:               # 停止条件2: 只剩 0 种 特征
            rt = rest_y.value_counts().idxmax()  # 返回占比大的 label
            self.assign_value(cn, rt)
            return 'stop'
        rest_type_y = pd.unique(rest_y)
        if len(rest_type_y) == 1:                # 停止条件1: 只剩 1种 label
            rt = pd.unique(rest_y)[0]
            self.assign_value(cn, rt)
            return 'stop'
        return 'not stop'

    def pruning(self,X,y, pruning_type = 'predicative pruning'):
        if pruning_type == 'predicative pruning':
            # 预剪枝
            pass
        elif pruning_type == 'post pruning':
            # 后剪枝
            pass
        return

    # def discretize(self,X,y,continual_feature = None):
    #     # 连续属性离散化, 根据各个划分点的信息增益大小
          # 还在更新当中！
    #     for feature in continual_feature:
    #         # feature: the name of one continual attribute
    #         fv = X[feature]
    #         sort_fv = np.sort(fv)   # 从小到大排序的 feature value
    #         ta_value, tsf_ = None, None
    #         # ta_value: 得到的划分中值 Ta
    #         # tsf_:     得到的划分后的序列
    #         for i, value in sort_fv[:-1]:
    #             ta = (sort_fv[i] + sort_fv[i + 1]) / 2          # 中间值
    #             tsf = np.array(list(map(lambda x: 1 if (x > ta) else 0, fv)))
    #             tvi = 00 # 计算的 Information gain
    #             if i == 0 or tvi >= ta_value:
    #                 tsf_ = tsf      # 将本次计算得到的 离散化后的值赋给 tsf_
    #                 ta_value = tvi  # 将本次计算得到的 ta 赋给 ta_value
    #         X[feature] = tsf_
    #     return


    def fit(self,X,y,depth = 0, cn = None, k_bf = '*'):
        # 连续属性离散化放在哪个位置 ?
        clm = list(X.columns)   # 目前 X 剩余 feature 种类
        if self.stop_criterion(y, clm, cn) == 'stop': # 检查是否满足 停止条件1, 2
            return
        E, ig_set = self.entropy(y), [] # 计算信息熵, 初始化 储存信息增益的 container
        for feature in clm:
            fv= X.loc[:, feature]
            unique_fv = pd.unique(fv)    # feature 特征值的种类, 如:长度有 A, B, C 3种特征
            idx_set = [X.index[fv == i] for i in unique_fv]     # 各类特征下标集合
            ig_set.append(self.compute([np.array(y[idx]) for idx in idx_set], E))

        sf = clm[np.argmax(np.array(ig_set))]   # split_feature 返回 IG 最大的特征
        clm.remove(sf)  # 除去 sf
        ls = X.loc[:, sf]   # label_set
        unique_ls = pd.unique(ls)       # 决定划分的特征 sf 的 feature value 的种类
        idx_feature = [X.index[ls == i] for i in unique_ls]     # 上行各个种类的下标

        if self.signal != 0:
            k_bf = self.assign_value(cn, sf)  # lf 上个划分的 feature ? 变为划分的 feature
        elif self.signal == 0:
            self.y_set = np.unique(y)       # 为self.y_set 赋值
            self.signal = self.signal + 1
        depth = depth + 1
        for n, idx_each in enumerate(idx_feature):
            cn = [depth, k_bf, sf, unique_ls[n], '?']
            self.store(cn)
            self.fit(X[clm].loc[idx_each], y[idx_each], depth = depth, cn = cn)
        return


    def store(self, cn):
        # 传入 cn 将 cn 保存到当前的 self.container 中
        layer, k_bf, k_now, fv, k_nex = cn[0], cn[1], cn[2], cn[3], cn[4]
        # layer: 当前划分的深度
        # k_bf:  上一个划分的特征
        # k_now: 当前划分的特征
        # fv:    该特征对应的特征值
        # k_nex: 下一个划分的特征
        key_now, value_next = (k_bf, k_now), (k_now, k_nex)
        # key_now (a,b): {1: (b,d)} 中的 (a,b)
        # value_next (b,d)
        if self.container == [] or len(self.container) < layer: # 若第一次划分某一个深度
            self.container.append({key_now: {fv: value_next}})
        else:
            if key_now in self.container[layer - 1].keys():
                self.container[layer - 1][key_now][fv] = value_next
            else:
                self.container[layer - 1][key_now] = {fv: value_next}


    def predict(self, X_test):  # Checked √
        prediction = []
        for idx, sample in X_test.iterrows():
            feature = list(self.container[0].keys())[0]
            n = 0
            while True:
                trait = feature[1]
                value = sample[trait]
                result = self.container[n][feature][sample[trait]][1]
                if result in self.y_set:  # whether it arrives the end
                    prediction.append(result)
                    break
                else:
                    feature = (str(feature[0]) + str(feature[1]) + str(value),
                               self.container[n][feature][sample[trait]][1])
                    n = n + 1
        return np.array(prediction)

    def score(self,X_test, y_test): # Accuracy
        return np.sum(self.predict(X_test) == y_test) / len(y_test)

if __name__ == '__main__':
    data = pd.read_excel('D:\\tool\\Machine_Learning1\\com\\Data\\Datasets\\Dealt\\breast.xls')
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    # Sklearn 包中的决策树
    sklearn_tree = tree.DecisionTreeClassifier(random_state = 10).fit(X, y)
    # 自己写的决策树 mytree
    mytree = DT()
    mytree.fit(X, y)
    print('最终的 my tree 的container:')
    for i, _ in enumerate(mytree.container):
        print('第', i + 1, '层')
        for key in _:
            print('k:',key,'v:',_[key])
        print('-------------')
    print('sklearn决策树:')
    print(sklearn_tree.score(X, y))
    print('mytree决策树 :')
    print(mytree.score(X, y))

