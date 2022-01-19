import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class Standardization:
    def __init__(self, colunm, Name):
        self.Column = colunm
        self.Data = pd.read_csv(Name, names = self.Column)
        
        
    def Data_Standard_Trans(self):
        Standard_Temp = np.zeros((len(self.Data[self.Column[0]].to_numpy()), len(self.Column)-1))
        for i in range(len(self.Column)-1):
            Data_Raw = self.Data[self.Column[i]].to_numpy().astype('float')
            av = np.average(Data_Raw)
            st = np.std(Data_Raw)
            for j in range (len(Data_Raw)):
                Standard_Temp[j,i] = (float(Data_Raw[j]) - av)/(st)
        
        return Standard_Temp

    def SplitY(self):
        return self.Data.iloc[:, -1].values

    def Accurate(self, y_true, y_pred):
        acc =  np.sum(y_true == y_pred)/ len(y_true)
        return acc

    
            
            

