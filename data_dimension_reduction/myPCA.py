import numpy as np

class myPCA:
    @staticmethod
    def mediate(row_vectors:np.array):
        num,dim=row_vectors.shape
        mean=np.mean(row_vectors,axis=0)    #求每一列的平均值
        return row_vectors - mean

    @staticmethod
    def fit(row_vectors,dim_goal):
        row_vectors=myPCA.mediate(row_vectors)
        num,dim=row_vectors.shape
        if dim_goal>=dim:
            raise ValueError("目标维度应比样本向量维度低")
        cov_Mat = np.dot(row_vectors.T, row_vectors) / (num - 1)
        U, V = np.linalg.eigh(cov_Mat)
        U = U[::-1]
        for i in range(dim):
            V[i, :] = V[i, :][::-1]
        # 保留 dim_goal 维度的特征值作为主成分
        return np.dot(row_vectors, V[:, :dim_goal] )
