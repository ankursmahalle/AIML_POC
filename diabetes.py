# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:02:18 2024

@author: MCS
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='lbfgs', max_iter=10000000)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# DATA FOR PRED
dp=pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
print(dp.head())

sns.countplot(x='Outcome',data=dp)
plt.show
sns.distplot(dp['Insulin'])
sns.boxplot(x='Outcome',col='Insulin', kind='count',data=dp);

logreg=LogisticRegression()



X=dp.iloc[:,:8]
print(X.shape[1])


y=dp[["Outcome"]]

X=np.array(X)
y=np.array(y)


#logreg.fit(X,y.reshape(-1,))
#joblib.dump(logreg,"model1")