from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from Dim_RD import Dim_RD
from myPCA import myPCA
import os
print(os.getcwd())
with open("matteonormb.txt", "r") as f:
    raw_data=f.read()
# print(raw_data)
_data3D=np.array([list(map(float,line.split()[1:])) for line in raw_data.split('\n')])

data3D=_data3D.copy()
# zdata = data3D.transpose()[0]
# xdata = data3D.transpose()[1]
# ydata = data3D.transpose()[2]
p3D = plt.axes(projection='3d')
xdata,ydata,zdata = tuple(data3D.transpose())
p3D.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')
plt.show()

pca = PCA(copy=True,n_components = 2)
data2D_pca=pca.fit_transform(data3D)
plt.scatter(*tuple(data2D_pca.transpose() ),c= "red")
plt.title("sklearn PCA")
plt.show()

data2D_pca_mine=myPCA.fit(data3D,dim_goal=2)
plt.scatter(*tuple(data2D_pca_mine.transpose() ),c= "pink")
plt.title("my PCA")
plt.show()

data3D=_data3D.copy()
# print(data3D)
mds=MDS(n_components=2)
data2D_mds=mds.fit_transform(data3D)
# print(data2D_mds)
plt.scatter(*tuple(data2D_mds.transpose() ),c= "blue")
plt.title("sklearn MDS")
plt.show()


