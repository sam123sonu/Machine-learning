# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:14:11 2020

@author: sambit mohapatra
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv(r"E:\ML\Churn_Modelling.csv")

df = data.drop(['RowNumber','CustomerId',
                'Surname'],axis=1)

#Analysing the data

#checking for null values

df.isnull().sum().sum()
sns.distplot(df.CreditScore[df.Exited==0])
sns.distplot(df.CreditScore[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()


sns.distplot(df.Age[df.Exited==0])
sns.distplot(df.Age[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()


sns.distplot(df.EstimatedSalary[df.Exited==0])
sns.distplot(df.EstimatedSalary[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()


sns.distplot(df.Balance[df.Exited==0])
sns.distplot(df.Balance[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()


sns.distplot(df.Tenure[df.Exited==0])
sns.distplot(df.Tenure[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()

sns.distplot(df.NumOfProducts[df.Exited==0])
sns.distplot(df.NumOfProducts[df.Exited==1])
plt.legend(['Not-exited','Exited'])
plt.show()


#Plotting Categorical value count plots

sns.countplot(df.Gender)
plt.show()

sns.countplot(df.Gender[df.Exited==1])
plt.show()


sns.countplot(df.Geography)
plt.show()

sns.countplot(df.Geography[df.Exited==1])
plt.show()

sns.countplot(df.HasCrCard)
plt.show()

sns.countplot(df.HasCrCard[df.Exited==1])
plt.show()


sns.countplot(df.IsActiveMember)
plt.show()

sns.countplot(df.IsActiveMember[df.Exited==1])
plt.show()
df.columns

ip = df.drop(['CreditScore', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'EstimatedSalary',
       'Exited'],axis=1)

op = df.Exited

from sklearn.preprocessing import LabelEncoder

l1 = LabelEncoder()
l2 = LabelEncoder()

ip.Geography = l1.fit_transform(ip.Geography)
ip.Gender = l2.fit_transform(ip.Gender)

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op,test_size = 0.2)

#scaling

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtr,ytr)
print(model.score(xts,yts))

#for confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(xts)
print(confusion_matrix(yts,y_pred))
