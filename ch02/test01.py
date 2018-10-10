import matplotlib.pyplot as plt
import mglearn
import sys
X,y = mglearn.datasets.make_forge()
#mglearn.discrete_scatter()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:{}".format(X.shape))
#plt.plot(X,y)
plt.show()


X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
print("Sample counts per class:\n{}".format(cancer.data.shape))
print("Sample counts per clas:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))


from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape:{}".format(boston.data.shape))