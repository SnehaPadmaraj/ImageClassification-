import numpy as np
from sklearn.datasets import load_digits
data1 = load_digits()
data1.data
data1.target
data1.images.shape
data1.imagelen = len(data1.images)
print(data1.imagelen)
n = 1222 #no. of samples out of 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(data1.images[n])
plt.show()
data1.images[n]
x = data1.images.reshape((data1.imagelen, -1))
y = data1.target
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20)
xtrain.shape
xtest.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(xtrain,ytrain)
n = 1222
result = model.predict(data1.images[n].reshape((1,-1)))
plt.imshow(data1.images[n], cmap = plt.cm.gray_r, interpolation = 'nearest')
result
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, model.predict(xtest)))
model=SVC(kernel='sigmoid')
result
