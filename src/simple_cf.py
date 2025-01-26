# 简单的协同过滤算法
import math
from typing import Dict
import random
from tqdm import tqdm

import pandas as pd


def readDatas():
    path = "../ml-latest-small/ml-latest-small/ratings.csv"
    # 只读取0,1列
    datas = pd.read_csv(path, usecols=[0, 1])
    user_dict = dict()
    for d in datas.values:
        if d[0] in user_dict:
            user_dict[d[0]].add(d[1])
        else:
            user_dict[d[0]] = {d[1]}
    return user_dict



def readItemsDatas():
    path = "../ml-latest-small/ml-latest-small/ratings.csv"
    # 只读取0,1列
    datas = pd.read_csv(path, usecols=[0, 1])
    item_dict = dict()
    for d in datas.values:
        if d[1] in item_dict:
            item_dict[d[1]].add(d[0])
        else:
            item_dict[d[1]] = {d[0]}
    return item_dict

def getTrainsetAndTestset(dct: Dict):
    trainset, testset = dict(), dict()
    for uid in dct:
        testset[uid] = set(random.sample(list(dct[uid]), math.ceil(0.2 * len(dct[uid]))))  # 随机取0.2的样本
        trainset[uid] = dct[uid] - testset[uid]
    return trainset, testset


def knn(dataset: Dict, k: int):
    user_sims = {}
    for u1 in tqdm(dataset):
        ulist = []
        for u2 in dataset:
            if u1 == u2 or len(dataset[u1] & dataset[u2]) == 0: continue
            rate = cossim(u1, u2, dataset)
            ulist.append({"id": u2, 'rate': rate})
        # 取前k个
        user_sims[u1] = sorted(ulist, key=lambda u: u['rate'], reverse=True)[:k]
    return user_sims


# 余弦相似度计算
def cossim(s1, s2, dataset):
    return len(dataset[s1] & dataset[s2]) / (len(dataset[s1]) * len(dataset[s2])) ** 0.5


def get_recomedations(user_sims, o_set):
    recomendation = dict()
    for u in tqdm(user_sims):
        recomendation[u] = set()
        for sim in user_sims[u]:
            recomendation[u] |= o_set[sim['id']] - o_set[u]
    return recomendation

def get_recomedations_by_itemCF(item_sims,o_set):
    recomendation = dict()
    for u in tqdm(o_set):
        recomendation[u] = set()
        for item in o_set[u]:
            recomendation[u] |= set(i['id'] for i in item_sims[item]) - o_set[u]
    return recomendation

def precisionAndRecall(pre, test):
    p, r = 0,0
    for uid in test:
        t = len(pre[uid] & test[uid])
        p += t / (len(pre[uid]) + 1)
        r += t / (len(test[uid]) + 10)
    return p / len(test),r / len(test)



def play():
    odatas = readDatas()
    item_datas = readItemsDatas()
    trset, test = getTrainsetAndTestset(odatas)
    user_sims = knn(trset, 5)
    item_sims = knn(item_datas,5)
    # print(user_sims)
    pre_set = get_recomedations(user_sims, trset)
    pre_itemCF_set = get_recomedations_by_itemCF(item_sims,trset)
    p,r = precisionAndRecall(pre_set,test)
    pi,ri = precisionAndRecall(pre_itemCF_set,test)
    print(p,r)
    print(pi,ri)




# datas = readDatas()
# print(getTrainsetAndTestset(datas))
if __name__ == '__main__':
    play()
