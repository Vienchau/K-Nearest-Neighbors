import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from Standardization import Standardization


Columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
Standard = Standardization(Columns, "iris.data")
test = Standard.Data_Standard_Trans()
y= Standard.SplitY()
X_train, X_test, y_train, y_test = train_test_split(test, y, test_size= 0.2)
classifier = KNeighborsClassifier(n_neighbors=9, weights= 'distance')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = (Standard.Accurate(y_test, predictions))*100
print(accuracy)





