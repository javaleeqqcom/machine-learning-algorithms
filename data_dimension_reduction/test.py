import numpy as np
from myLLE import LocallyLinearEmbedding as LLE

import os
print(os.getcwd())
with open("matteonormb.txt", "r") as f:
    raw_data=f.read()
# print(raw_data)
_data3D=np.array([list(map(float,line.split()[1:])) for line in raw_data.split('\n')])
lle=LLE(2,3)
lle.fit_transform(_data3D)