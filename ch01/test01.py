import numpy as np

x = np.array([[1,2,3],[4,5,6]])
print(x)

from scipy import sparse
#创建一个二维NumPy数组，对角线为1，其余都为0
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

#将NumPy数组转换为CSR格式的SciPy稀疏矩阵
#只保存非零元素




sparse_matrix = sparse.csc_matrix(eye)
print("\nSciPy sparse matrix:\n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
print("COO representation :\n{}".format(eye_coo))
#matplotlib inline
import matplotlib.pyplot as plt

#在-10和10 之间生成一个数列，共100个数

x = np.linspace(-10,10,100)
#用正弦函数创建第二个数组
y= np.sin(x)

#plt函数绘制一个数组关于另外一个数组的折线图
plt.plot(x,y,marker="x")
plt.show()

import pandas as pd

from IPython.display import display

data = {'Name':["John","Anna","Peter","Linda"],'Location':["New York","Paris","Berlin","London"],'Age':[24,13,53,33]}
data_pandas = pd.DataFrame(data)
#IPython.display 可以在Jupyter Notebook中打印出“美观的"DataFrame
display(data_pandas)

#选择年龄大于30的所有行
display(data_pandas[data_pandas.Age > 30])

import sys
print("Python version :{}".format(sys.version))

import pandas as pd
print("matplotlib version:{}".format(pd.__version__))

import  matplotlib
print(matplotlib.__version__)


import numpy as np
print(np.__version__)

import scipy as sp
print(sp.__version__)

import IPython
print(IPython.__version__)
import sklearn

print(sklearn.__version__)