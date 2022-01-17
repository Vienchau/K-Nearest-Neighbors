import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class KNN:
	def __init__(self, Data):
		self.Data = Data 
		#self.ACC = self.Accurate
		print(self.Data.head())



	def Data_splitXY(self):
		_X = self.Data.iloc[:, :-1].values
		_y = self.Data.iloc[:, -1].values
		le = LabelEncoder()
		_y = le.fit_transform(_y)
		X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size= 0.2)

		return X_train, X_test, y_train, y_test

	def Accurate(self, y_true, y_pred):
		acc =  np.sum(y_true == y_pred)/ len(y_true)
		return acc


	def Data_Classifier(self, _X_train, _X_test, _y_train, _y_test, _n_neighbors):
		classifier = KNeighborsClassifier(n_neighbors=_n_neighbors, weights= 'distance')
		classifier.fit(_X_train, _y_train)
		predictions = classifier.predict(_X_test)
		accu = (self.Accurate(_y_test, predictions) * 100)
		return accu 

	def Data_Classifier_with_Standard(self, _X_train2, _X_test2, _y_train2, _y_test2, _n_neighbors):
		sc = StandardScaler()
		_X_train2 = sc.fit_transform(_X_train2)
		_X_test2 = sc.transform(_X_test2)
		classifier = KNeighborsClassifier(n_neighbors=_n_neighbors,weights= 'distance')
		classifier.fit(_X_train2, _y_train2)
		predictions2 = classifier.predict(_X_test2)
		accu = (self.Accurate(_y_test2, predictions2) * 100)
		return accu 

	def Data_Classifier_with_Nomalize(self, _X_train2, _X_test2, _y_train2, _y_test2, _n_neighbors):
		scaler = MinMaxScaler(feature_range=(0,1))
		_X_train2[:, :2] = scaler.fit_transform(_X_train2[:, :2])
		_X_test2[:, :2] = scaler.transform(_X_test2[:, :2])
		classifier = KNeighborsClassifier(n_neighbors=_n_neighbors, weights= 'distance')
		classifier.fit(_X_train2, _y_train2)
		predictions2 = classifier.predict(_X_test2)
		accu = (self.Accurate(_y_test2, predictions2) * 100)
		return accu 

