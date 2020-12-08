# author: SZB time:2020/4/26

import numpy as np
import gzip
import struct
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import scipy.io as scio
from scipy.stats import multivariate_normal
'''

EM算法的核心理解

E步
由初始值αk, 均值μk， 方差Σk，
计算出N个样本X,对应K个高斯模型的概率γjk, j 从1到N表示样本，k表示第k个模型
然后对K个概率求和sum(γjk)，k从1到K，需要注意此步有可能出现求和为0的情况
接着计算第k个模型对第j个样本xj的响应度γ(jk)= γjk/sum(γjk)

M步
根据响应度更新参数αk+1, 均值μk+1， 方差Σk+1
其中αk+1 =sum[γ(jk) ]/N            对j求和
μk+1 = sum(γ(jk)*xj) / sum(γ(jk))
Σk+1 = sum(γ(jk)*(xj-μk)**2)/sum(γ(jk))

'''

# 利用PCA对读取的样本进行特征提取
def feature_PCA(images,c_num=0):
    if 0==c_num:
        pca = PCA(copy=True,n_components = 0.95,whiten=True)
    else:
        pca = PCA(copy=True,n_components = c_num,whiten=True)
    # pca = PCA(copy=True,n_components = 'mle',whiten=True)
    data = pca.fit_transform(images)
    return data

# 定义EM算法的概率模型函数
def Gaussain_prob(X, mu, cov):
    norm = multivariate_normal(mean = mu, cov = cov)
    return norm.pdf(X)

# 参数初始化
def init_params(X, K):
    N, D = X.shape
    mu = np.random.rand(K, D)                        # shape(K,D)
    cov = np.array([np.eye(D)]*K)  # shape(K,D,D)
    alpha = np.array([1.0/K]*K)                        # shape(K,)
    return N, D, K,  mu, cov, alpha

# E步
def E_step(X, N, K,  mu, cov, alpha):
    gamma_prob = np.mat(np.zeros((N, K)))
    for k in range(K):
        gamma_prob[:, k] = alpha[k] * Gaussain_prob(X, mu[k], cov[k]).reshape(-1, 1)
    sum_gamma_prob = np.sum(gamma_prob, axis = 1)
    gamma_prob = gamma_prob/(sum_gamma_prob)    # 防止除零
    return gamma_prob

def M_step(X, K, D, N, gamma_prob):
    new_mu = np.zeros([K, D])
    new_cov = []
    new_alpha = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma_prob[:, k])
        new_mu[k, :] = np.dot(gamma_prob[:, k].T, X) / Nk
        diff = np.mat(X - new_mu[k])
        cov = np.array(diff.T * np.multiply(diff, gamma_prob[:, k]) / Nk)   # 有可能出现非正定协方差阵
        new_cov.append(cov)
        new_alpha[k] = Nk / N
    new_cov = np.array(new_cov)
    return new_mu, new_cov, new_alpha

def predict(gamma_prob, K):
    prob = tuple(gamma_prob.argmax(axis = 1).T.tolist()[0])

    count_num = []
    for i in range(K):
        num = prob.count(i)
        count_num.append(num)
    print(count_num)
    print(np.max(count_num))
    return np.max(count_num)

def sort_lable(data, lable):
    data_sorted = {}
    for i in set(lable):
        data_sorted[i] = []
    L = len(lable)
    for l in range(L):
        data_sorted[lable[l]].append(data[l,:])
    return data_sorted, set(lable)

if __name__ == '__main__':
    mnist = scio.loadmat('../datasets/mnist-original.mat')
    Ximage_data, lable = mnist["data"].transpose(), np.array(mnist["label"],dtype='int32').flatten()

    print(lable.shape)
    print(lable)

    total_num=int(len(lable))    #总样本数
    # total_num=500    #总样本数
    train_num=int(0.7*total_num)#训练样本数

    index = np.arange(total_num)  # 生成下标
    np.random.shuffle(index)        #打乱下标
    #训练集
    train_image=Ximage_data[index[:train_num],:]
    train_lable=lable[index[:train_num]]
    print(set(lable))
    print(train_num)
    print(max(index[:train_num]))
    print(set(train_lable))
    #测试集
    test_image=Ximage_data[index[train_num:],:]
    test_lable=lable[index[train_num:]]

    #训练集PCA提取特征
    train_data=feature_PCA(train_image)
    print(train_data.shape)
    #测试集PCA提取特征 (要保证特征维度一致性)
    test_data=feature_PCA(test_image,c_num=train_data.shape[1])
    print(test_data.shape)

    N, D, K, mu, cov, alpha = init_params(train_data, K = 10)       # 初始化参数,已知MNIST数据集类别为10,设K=10
    iterate_times = 50
    test_data, lable_set = sort_lable(test_data, test_lable)
    print(test_lable)
    print(lable_set)
    for i in range(iterate_times):
        s = 0
        print('第%s步' %i)
        gamma_prob = E_step(train_data, N, K, mu, cov, alpha)
        mu, cov, alpha = M_step(train_data, K, D, N, gamma_prob)
        for m in range(cov.shape[1]):	# 处理运算过程中出现的奇异阵问题，算法收敛可能会变慢，对准确率影响不大
            cov[:, m, m] += 1e-9
        for l in lable_set:
            gamma_prob = E_step(test_data[l], len(test_data[l]), K, mu, cov, alpha)
            s += predict(gamma_prob, K)
        result_prob = s / len(test_lable)
        print('预测准确率%f' %result_prob)
