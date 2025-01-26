import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def readDatas():
    path = "../ml-latest-small/ml-latest-small/ratings.csv"
    # 只读取0,1列
    datas = pd.read_csv(path, usecols=[0, 1, 2])
    return  datas

def splitTrainSetAndTestSet(odatas,frac):
    testset = odatas.sample(frac=frac,axis=0)
    trainset = odatas.drop(index=testset.index.values.tolist(),axis=0)
    return trainset,testset

class LFM:
    def __init__(self,dataset,factors,epoch,lr,lamda):
        self.dataset = dataset
        self.factors = factors
        self.userList,self.itemList = self.__getListMap()
        self.epoch = epoch
        self.lr = lr
        self.lamda = lamda
        self.p = pd.DataFrame(np.random.randn(len(self.userList),factors),index=self.userList)
        self.q = pd.DataFrame(np.random.randn(len(self.itemList), factors), index=self.itemList)
        self.bu = pd.DataFrame(np.random.randn(len(self.userList)), index=self.userList)
        self.bi = pd.DataFrame(np.random.randn(len(self.itemList)), index=self.itemList)

    def __getError(self,r,pu,qi,bu,bi):
        return r-self.__prediction(pu,qi,bu,bi)
    def __prediction(self,pu,qi,bu,bi):
        return (np.dot(pu,qi.T) + bu + bi)[0]

    def __getListMap(self):
        userSet,itemSet = set(),set()
        for d in self.dataset.values:
            userSet.add(int(d[0]))
            itemSet.add(int(d[1]))
        userList = list(userSet)
        itemList = list(itemSet)
        return userList,itemList

    def fit(self):
        for e in tqdm(range(self.epoch)):
            for d in self.dataset.values:
                u,i,r = d[0],d[1],d[2]
                error = self.__getError(r,self.p.loc[u],self.q.loc[i],self.bu.loc[u],self.bi.loc[i])
                self.p.loc[u] += self.lr * (error * self.q.loc[i] - self.lamda * self.p.loc[u])
                self.q.loc[i] += self.lr * (error * self.p.loc[u] - self.lamda * self.q.loc[i])
                self.bu.loc[u] += self.lr * (error - self.lamda * self.bu.loc[u])
                self.bi.loc[i] += self.lr * (error - self.lamda * self.bi.loc[i])
    def __RMSE(self,a,b):
        return  (np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5

    def testRMSE(self,testSet):
        y_true,y_hat = [],[]
        for d in tqdm(testSet.values):
            user = int(d[0])
            item  = int(d[1])
            if user in self.userList and item in self.itemList:
                hat = self.__prediction(self.p.loc[user],self.q.loc[item],self.bu.loc[user],self.bi.loc[item])
                y_hat.append(hat)
                y_true.append(d[2])
        rmse = self.__RMSE(y_true,y_hat)
        return rmse
    def save(self,path):
        with open(path,"wb+") as f:
            pickle.dump(self,f)
    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return pickle.load(f)
def play():
    factors = 100 # 隐因子数量
    epochs = 1000
    lr = 0.001
    lamda = 0.1

    model_path = './model/lfm.model'
    trainset,testset = splitTrainSetAndTestSet(readDatas(),0.2)
    # lfm = LFM.load(model_path) # 如果有模型的话 可以直接加载
    lfm = LFM(trainset,factors,epochs,lr,lamda)
    lfm.fit()
    lfm.save(model_path)
    rmse_test = lfm.testRMSE(testset)
    rmse_train = lfm.testRMSE(trainset)
    print(rmse_train)
    print(rmse_test)

if __name__ == '__main__':
    play()