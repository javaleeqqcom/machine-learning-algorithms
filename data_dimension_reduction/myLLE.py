import numpy as np
from sklearn import neighbors

class LocallyLinearEmbedding:
    # p>=1 ,p=2为欧式距离 ; p=1为曼哈顿距离 ；p=inf为最值距离
    def __init__(self,n_components,n_neighbors,p=2):
        self.n_neighbors=n_neighbors    #取n_neighbors个邻近样本，记为k
        self.n_components=n_components  #降为 n_neighbors 维，记为 d
        self.p=p

    def fit_transform(self,X:np.array):
        m,n=X.shape # m个n维向量，m个点
        kdtree=neighbors.KDTree(X, leaf_size=40)
        # 求K近邻，inds是一个m行k+1列向量。inds[i,?]=j 表示X[j]是X[i]的一个邻近向量，特别地 inds[i,0]==i 因为自己离自己最近。
        inds=kdtree.query(X,k=self.n_neighbors+1,return_distance=False)
        # print("inds: ",inds)

