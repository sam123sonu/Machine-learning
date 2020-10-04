# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:51:24 2020

@author: sambit
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv(r"E:\Ml\Churn_Modelling.csv")

df = data.drop(['RowNumber','CustomerId',
                'Surname'],axis=1)

#Analysing the data

#checking for null values

df.isnull().sum().sum()

#freq distribution of continuous values vs exited

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


#features that affect exiting tendency :

# Age, Gender, Geography, IsActiveMember

#df.columns

ip = df.drop(['CreditScore', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'EstimatedSalary',
       'Exited'],axis=1)

op = df.Exited


#OneHotEncoding the categorical/string columns
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ct = ColumnTransformer(transformers = 
                       (['Geography',OneHotEncoder(),[0]],
                        ['Gender',OneHotEncoder(),[1]],
                        ['IsActiveMember',OneHotEncoder(),[3]]),
                       remainder='passthrough')

ip = ct.fit_transform(ip)


from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op, test_size=0.2)


#scaling the data = standerization


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xtr,ytr)
model.score(xts,yts)

#confusion matrix


'''
[[ true positive  false positive]
[ false negative  true negative]]
'''


from sklearn.metrics import confusion_matrix

y_pred = model.predict(xts)

print(confusion_matrix(yts, y_pred))



































