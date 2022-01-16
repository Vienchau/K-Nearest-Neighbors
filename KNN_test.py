import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from KNN  import  KNN

columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
Iris_Data = data_df = pd.read_csv("./iris.data", names = columns)
KNN_test = KNN(Iris_Data)

X_train, X_test, y_train, y_test = KNN_test.Data_splitXY()

acc1 = KNN_test.Data_Classifier(X_train, X_test, y_train, y_test, 11)
print(acc1)

acc2 = KNN_test.Data_Classifier_with_Standard(X_train, X_test, y_train, y_test, 11)
print(acc2)

acc3 = KNN_test.Data_Classifier_with_Nomalize(X_train, X_test, y_train, y_test, 11)
print(acc3)
