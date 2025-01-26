import os

import numpy as np
from mxnet import nd,gluon,autograd

base_path = "../ml-100k/ml-100k"
train_path = os.path.join(base_path,"ua.base")
test_path = os.path.join(base_path,"ua.test")
user_path = os.path.join(base_path,"u.user")
item_path = os.path.join(base_path,'u.item')
occupation_path = os.path.join(base_path,"u.occupation")

# ＞ 3 的为1 否则为0
def parse(r):
    return 1.0 if r > 3 else 0.0

def __read_rating_datas(path):
    dataset = {}
    with open(path,"r") as f:
        for line in f.readlines():
            d = line.strip().split("\t")
            dataset[(int(d[0]),int(d[1]))] = [parse(int(d[2]))]
    return dataset

def __read_item_hot():
    items = {}
    with open(item_path,'r',encoding="ISO-8859-1") as f:
        for line in f.readlines():
            d = line.strip().split('|')
            items[int(d[0])] = np.array(d[5:],dtype="float32")
    return items
def __read_occupation_hot():
    occupations = {}
    with open(occupation_path,"r") as f:
        names = f.read().strip().split("\n")
    length = len(names)
    for i in range(length):
        l = np.zeros(length,dtype="float32")
        l[i] = 1
        occupations[names[i]] = l
    return  occupations

def __read_user_hot():
    user = {}
    gender_dict = {"M":1,"F":0}
    occupation_dict = __read_occupation_hot()
    with open(user_path,'r') as f:
        for line in f.readlines():
            d = line.strip().split('|')
            a =np.array([int(d[1]),gender_dict[d[2]]])
            user[int(d[0])] = np.append(a,occupation_dict[d[3]])
    return  user

def read_dataSet(user_dict,item_dict,path):
    X,Y = [],[]
    ratings = __read_rating_datas(path)
    for k in ratings:
        X.append(np.append(user_dict[k[0]],item_dict[k[1]]))
        Y.append(ratings[k])
    return X,Y
def read_data():
    user_dict = __read_user_hot()
    item_dict = __read_item_hot()
    trainX,trainY = read_dataSet(user_dict,item_dict,train_path)
    testX,testY = read_dataSet(user_dict,item_dict,test_path)
    return  trainX,trainY,testX,testY

epochs = 10000
batchSize = 1000
features = 42

def sigmoid(x):
    return  1/(1+nd.exp(-x))

def dataIter(bath_size,trainX,trainY):
    # 将 numpy 数组转换为 mxnet.ndarray.NDArray
    trainX = nd.array(trainX)
    trainY = nd.array(trainY)
    dataset = gluon.data.ArrayDataset(trainX,trainY)
    train_data_iter = gluon.data.DataLoader(dataset,bath_size,shuffle=True)
    return train_data_iter

class Net:
    def __init__(self):
        self.w0 = nd.random_normal(shape=(1,1),scale = 0.01,dtype='float32')
        self.w = nd.random_normal(shape=(features,1),scale = 0.01,dtype='float32')
        self.parms = [self.w0,self.w]
        for parm in self.parms:
            parm.attach_grad()

    def fit(self,x):
        z = nd.dot(x,self.w)
        return sigmoid(z)

    def SGD(self,lr):
        for parm in self.parms:
            parm[:] = parm - lr * parm.grad

def train(trainX,trainY):
    train_data_iter = dataIter(batchSize,trainX,trainY)
    sigmoidBCEloss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid = True)
    lr = 0.00001
    net = Net()
    for e in  range(epochs):
        total_loss = 0
        for x,y in train_data_iter:
            with autograd.record():
                y_hat = net.fit(x)
                loss = sigmoidBCEloss(y_hat,y)
                loss.backward()
            net.SGD(lr)
            total_loss += nd.sum(loss).asscalar()
        print(f"epoch:{e},loss:{total_loss/len(trainY)}")

if __name__ == '__main__':
    trainX, trainY, testX, testY = read_data()
    train(trainX,trainY)


