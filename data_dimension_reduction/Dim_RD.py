import numpy as np

class Dim_RD:
    @staticmethod
    def mediate(row_vectors:np.array):
        num,dim=row_vectors.shape
        mean=np.mean(row_vectors,axis=0)    #求每一列的平均值
        return row_vectors - mean

    @staticmethod
    def pca_fit(row_vectors,dim_goal):
        row_vectors=Dim_RD.mediate(row_vectors)
        num,dim=row_vectors.shape
        if dim_goal>=dim:
            raise ValueError("目标维度应比样本向量维度低")
        cov_Mat = np.dot(row_vectors.T, row_vectors) / (num - 1)
        U, V = np.linalg.eigh(cov_Mat)
        U = U[::-1]
        for i in range(dim):
            V[i, :] = V[i, :][::-1]
        # 保留 dim_goal 维度的特征值作为主成分
        return row_vectors.dot( V[:, :dim_goal] )
    # @staticmethod
    # def MDS(D, d):
    #     D = np.asarray(D)
    #     DSquare = D ** 2
    #     totalMean = np.mean(DSquare)
    #     columnMean = np.mean(DSquare, axis=0)
    #     rowMean = np.mean(DSquare, axis=1)
    #     B = np.zeros(DSquare.shape)
    #     for i in range(B.shape[0]):
    #         for j in range(B.shape[1]):
    #             B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    #     eigVal, eigVec = np.linalg.eig(B)  # 求特征值及特征向量
    #     # 对特征值进行排序，得到排序索引
    #     eigValSorted_indices = np.argsort(eigVal)
    #     # 提取d个最大特征向量
    #     topd_eigVec = eigVec[:, eigValSorted_indices[:-d - 1:-1]]  # -d-1前加:才能向左切
    #     X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d - 1:-1])))

