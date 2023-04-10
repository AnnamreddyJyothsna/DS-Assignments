# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:42 2023

@author: ANNAMREDDY  JYOTHSNA
"""

# Question -1

#----------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("delivery_time.csv")
df.head()
df.tail()

df.plot(kind='box')
df.isnull().sum()
df.shape

x=df[["Sorting Time"]]
y=df["Delivery Time"]
import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='k')
plt.plot(df["Sorting Time"],Y_pred,color="r")
plt.show()
# log transformation

df['Sorting Time'] = np.log(df['Sorting Time'])
sns.distplot(df['Sorting Time'])
fig = plt.figure()

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)

Y_pred = LR.predict(x)
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(y,Y_pred)
print("Mean Squarred Error:", mse.round(3))
print("Root Mean Squarred Error:", np.sqrt(mse).round(3))
print(" Cube root:", np.cbrt(mse).round(3))


z = int(input())
t = np.array([[z]])
y=(LR.predict(t)).round(2)
print("output for new x value",y)

# To add the predicted value to the original dataset

b = {"Sorting Time":z,"Delivery Time":y[0]}
(pd.DataFrame(b,index=[0]))
df=df.append(b,ignore_index=True)
df


x= df[["Sorting Time"]]
y=df["Delivery Time"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
Ypred = Lr.predict(x)

# showing new predicted value in scatter plot

import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='k')
plt.plot(df["Sorting Time"],Ypred,color="g")
plt.show()


#------------------------------------------------------------
# Question-2
#-------------------------------------------------------


import numpy as np
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df.head()
df.tail()

df["YearsExperience"].plot(kind="box")
df["Salary"].plot(kind="box")

x=df[["YearsExperience"]]
y=df["Salary"]

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)


Y_pred = LR.predict(x)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,Y_pred)
print("Mean Squarred Error:", mse.round(3))
print("Root Mean Squarred Error:", np.sqrt(mse).round(3))
print("cube root:", np.cbrt(mse).round(3))


import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"],y=df["Salary"],color='green')
plt.plot(df["YearsExperience"],Y_pred,color="red")
plt.show()

# log transformation

df['YearsExperience'] = np.log(df['YearsExperience'])
sns.distplot(df['YearsExperience'])
fig = plt.figure()


a=int(input())
t = np.array([[a]])
y=(Lr.predict(t)).round(2)
print("output for new x value",y)

# Add the predicted value to the original dataset

c = {"YearsExperience":a,"Salary":y[0]}
(pd.DataFrame(c,index=[0]))
df=df.append(c,ignore_index=True)
df

x= df[["YearsExperience"]]
y=df["Salary"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
ypred = Lr.predict(x)

# Showing new predicted value in the scatter plot

import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"],y=df["Salary"],color='k')
plt.plot(df["YearsExperience"],ypred,color="g")
plt.show()