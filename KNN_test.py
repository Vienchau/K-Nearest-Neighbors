import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from KNN  import  KNN

columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
Iris_Data = pd.read_csv("iris.data", names = columns)
KNN_test = KNN(Iris_Data)

Raw = np.zeros((5,4))

for i in range (5):
    X_train, X_test, y_train, y_test = KNN_test.Data_splitXY()

    acc1 = KNN_test.Data_Classifier(X_train, X_test, y_train, y_test, 9)
    Raw[i,0] = i
    Raw[i,1] = acc1
    acc2 = KNN_test.Data_Classifier_with_Standard(X_train, X_test, y_train, y_test, 9)
    Raw[i,2] = acc2
    acc3 = KNN_test.Data_Classifier_with_Nomalize(X_train, X_test, y_train, y_test, 9)
    Raw[i,3] = acc3


print(acc1)
print(acc2)
print(acc3)

plt.figure(figsize= (20,3), dpi= 100)
plt.plot(Raw[:,0], Raw[:,1], label = 'Raw')
plt.plot(Raw[:,0], Raw[:,2], label = 'Standardization')
plt.plot(Raw[:,0], Raw[:,3], label = 'Normalization')


plt.title('3 times test')
plt.xlabel('test')
plt.ylabel('Accuracy')
plt.legend(loc ='best')
plt.grid() 
plt.show()







