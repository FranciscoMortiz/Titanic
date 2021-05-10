import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import os 
from pandas.api.types import CategoricalDtype
#import helper_functions

sns.set_theme()

df = pd.read_csv("train.csv") #Reads data
nans = df.isna().sum() #Summary of NaN values in data
df = df.drop("Cabin",axis=1) # Cabin column is dropped because it has a lot of NaNs
df =  df.drop("Name",axis=1) # Name column is dropped since it has no relevance
df =  df.drop("Ticket",axis=1) # Ticket column is dropped since it has no relevance
df.interpolate(inplace=True) # perform linear interpolation to fill the NaNs (in Age column)
df.Embarked.fillna(df.Embarked.mode()[0], inplace=True) # replace NaNs in Embarked column with its mode

#* Change data type of some columns
cat_type = CategoricalDtype(categories=[3,2, 1], ordered=True)
df = df.astype({"Survived":"category", "Sex":"category", "Embarked":"category", "Pclass": cat_type})

#* Summary of data
describe = df.describe() #Summary of data
info = df.info()

#* VISUALIZATION

"""plt.subplot(121) 
df.Age.plot(kind="box",title="Age")
plt.subplot(122)
df.Fare.plot(kind="box",title="Fare")
plt.show()""" #Here we can see that Fare and Age have a lot of outliers
            
Q1 = df[["Age","Fare"]].quantile(0.25)
Q3 = df[["Age","Fare"]].quantile(0.75)
IQR = Q3 - Q1
outliers = (df[["Age","Fare"]] < (Q1 - IQR * 1.5)) | (df[["Age","Fare"]] > (Q3 + IQR * 1.5))

#Here the outliers are removed 
ageind=df[outliers.Age].index.to_list() #indexes of outliers in age
fareind= df[outliers.Fare].index.to_list() #indexes of outliers in fare
dropind = fareind + ageind; dropind = pd.Series(dropind) #combine the indexes of both and cast to series
df2 = df.drop(dropind.unique())         
#Comparison between boxplot with and without outliers
plt.subplot(221)
df.Age.plot(kind="box", title = "Outliers")
plt.subplot(222)
df2.Age.plot(kind="box",  title = "No-outliers")
plt.subplot(223)
df.Fare.plot(kind="box",  title = "Outliers")
plt.subplot(224)
df2.Fare.plot(kind="box",  title = "No-outliers")
plt.show()

print(info)
# Univariate Analysis 



"""sns.scatterplot(x="Age", y="Fare", data=df)
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



#* FURTHER DATA CLEANING SHOULD BE CONSIDERED

print(df.describe())

