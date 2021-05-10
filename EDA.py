import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import os 
#import helper_functions

sns.set_theme()

df = pd.read_csv("train.csv") #Reads data
nans = df.isna().sum() #Summary of NaN values in data
df =df.drop("Cabin",axis=1) # Cabin column is dropped because it has a lot of NaNs
df.interpolate(inplace=True) # perform linear interpolation to fill the NaNs (in Age column)
df.Embarked.fillna(df.Embarked.mode()[0], inplace=True) # replace NaNs in Embarked column with its mode
describe = df.describe() #Summary of data


#Boxplots to identify outliers
#plt.figure("")

"""plt.subplot(121)
df.Age.plot(kind="box",title="Age")
plt.subplot(122)
df.Fare.plot(kind="box",title="Fare")
plt.show() 

sns.scatterplot(x="Age", y="Fare", data=df)
#plt.show() 

sns.displot(data=df, x="Age", kde=True)
#plt.show()

sns.displot(data =df,x="Fare", kde=True)
#plt.show()"""

""" There are outliers in AGE and FARE, we will not eliminate them since they're
    important data, so we will use the minkowski error at training time to kreduce
    the impact of the outliers in the model"""

"""
plt.figure("Two")
sns.boxplot(data=df, x='Survived', y='Fare')

plt.show() """

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - IQR * 1.5)) | (df > (Q3 + IQR * 1.5))

#Here the outliers are removed 
df2= df.loc[~outliers.Age,:]
df2 = df.loc[~outliers.Fare,:]

#* FURTHER DATA CLEANING SHOULD BE CONSIDERED

