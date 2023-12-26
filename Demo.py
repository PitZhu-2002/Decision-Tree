import collections
import math
import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self,container=[]):
        self.container = container
        self.signal = 0

    def entropy(self,feature_y):
        e = 0
        unique_elements, counts = np.unique(feature_y, return_counts = True)
        count_dict = dict(zip(unique_elements, counts))
        for value in count_dict.values():
            p = value / len(feature_y)
            e = e - p * math.log2(p)
        # 返回 信息熵
        return e

    def compute_ig(self,split_y, H):
        # split_y 划分出的样本
        # H 划分前的 Entropy
        size = sum([len(part) for part in split_y])
        for feature in split_y:
            H = H - self.entropy(feature) * len(feature) / size
        return H
    def compute_igr(self,split_y, H):
        # Information game ratio
        pass
    def compute_gini(self):
        pass

    def put_value(self,cn,rt):
        self.container[cn[0] - 1][cn[1]][cn[2]] = rt

    def fit(self,X,y,depth = 0,cn = None):
        clm = list(X.columns)
        if len(pd.unique(y)) == 1 or len(clm) == 1:
            rt = pd.unique(y)[0]
            self.put_value(cn,rt)
            return
            # print('layer:',depth)
            # print('return label:',rt)
            # print('container_after:', np.array(self.container))
            # print('end ---------------------------------')
        E = self.entropy(y)
        ig_set = []
        for feature in clm:
            unique_elements = pd.unique(X.loc[:,feature])   # 返回下标
            idx_set = [X.index[X.loc[:,feature] == i] for i in unique_elements]
            ig_set.append(self.compute_ig([np.array(y[idx]) for idx in idx_set],E))
        split_feature = clm[np.argmax(np.array(ig_set))]    # greater than 1 ?
        clm.remove(split_feature)
        label_set = pd.unique(X.loc[:,split_feature])
        idx_feature = [X.index[X.loc[:,split_feature] == i] for i in label_set]
        if self.signal == 0:
            self.signal += 1
        if self.signal != 0:
            self.put_value(cn, split_feature)

        # print('layer  :',depth)
        # print('feature:',split_feature)
        depth  += 1
        #print('extend ****************************')
        for n,idx_each in enumerate(idx_feature):
            cn = [depth,split_feature,label_set[n],'?']
            self.store(cn)
            # print('上层feature:', split_feature)
            # print('value:', label_set[n])
            self.fit(X[clm].loc[idx_each], y[idx_each], depth = depth, cn = cn)
        return

    def store(self,cn):
        # layer: depth
        # key: 上层 feature
        # key1: value
        # value1: next feature/ return label
        layer, key, key1, value1 = cn[0],cn[1],cn[2],cn[3]
        if self.container == [] or len(self.container) < layer:
            self.container.append({key: {key1: value1}})
        else:
            if key == self.container[layer - 1].keys() or key in self.container[layer - 1].keys():
                self.container[layer - 1][key][key1] = value1
            else:
                self.container[layer - 1][key] = {key1: value1}

    def predict(self, X_test):
        prediction = []
        for idx, sample in X_test.iterrows():
            feature = list(self.container[0].keys())[0]
            n = 0
            while True:
                feature = self.container[n][feature][sample[feature]]
                n = n + 1
                if feature == 'black' or feature == 'white':
                    prediction.append(feature)
                    break
        return np.array(prediction)

    def score(self,X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict == np.array(y_test)) / len(y_predict)

if __name__ == '__main__':
    data = pd.read_excel('D:\\tool\\Machine_Learning1\\com\\Data\\Datasets\\Dealt\\fruit.xls')
    idx = np.array([i for i in range(data.shape[0])])
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    tree = DecisionTree()
    tree.fit(X,y)
    print(tree.score(X,y))















# 思路:
# 1. 看 “上层feature” + “layer” + "value:b"
# [
# layer 1: {d: {2: b, 1: a, 0: black}}
# layer 2: {b: {0: white, 1: a, 2: black}, a: {2: c, 1: black, 0: black}}
# layer 3: {a: {0: white, 1: black}, c: {0: white, 1: black}}
# ]
#classifier = split(X,y,depth,container=[],signal=0)



#split(X,y)
#split_y = np.array([1,1,1,1,1,0,0,0,0,0])
#split_y = np.array([[1,1,1,1,0],[0,0,0,0,1]])
#split_y = np.array([[1,1,1,1,0,0,0],[0,0,1]],dtype = object)
