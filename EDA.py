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


#* VISUALIZATION


#* __________________ Univariate Analysis _________________________

# Histogram of numerical variables
fig, axs = plt.subplots(2,2)
fig.suptitle("Histogram of numerical variables")
sns.histplot(ax= axs[0,0], data=df,x="Age",  kde=True)
sns.histplot(ax=axs[0,1], data=df,x="Fare",  kde=True)
sns.histplot(ax=axs[1,0], data=df,x="SibSp",  discrete=True)
sns.histplot(ax=axs[1,1], data=df,x="Parch",  discrete=True )
plt.show()

# Histogram of categorical variables
fig, axs = plt.subplots(2,2)
fig.suptitle("Histogram of categorical variables")
sns.countplot(ax= axs[0,0], data=df,x="Sex")
sns.countplot(ax=axs[0,1], data=df,x="Embarked")
sns.countplot(ax=axs[1,0],x="Pclass",data=df, order=[1,2,3])
sns.countplot(ax=axs[1,1],x="Survived",data=df)
plt.show()

#? Conclutions 
"""
- Most passengers have 0 siblings/spouses
- Most passengers have 0 parents/children
- Most passengers embarked at port S
- There were more men than women at the ship
- There were more 3rd class passengers than 2nd and 1st and this last 2 were very close
- There is an imbalance in data, a lot more people died than survived
"""

#* __________________ Bivariate Analysis _________________________

# Histogram of survival on different variables
fig, axs = plt.subplots(2,4)
fig.suptitle("Survival rate on different variables")
sns.countplot(ax= axs[0,0], data=df,x="Sex", hue="Survived")
sns.countplot(ax=axs[0,1], data=df,x="Embarked", hue="Survived")
sns.countplot(ax=axs[0,2],x="Pclass",data=df, order=[1,2,3], hue="Survived")
sns.histplot(ax=axs[0,3], x="Age", data=df, hue="Survived", multiple="dodge", bins=5)
sns.histplot(ax=axs[1,0], data=df,x="Fare", hue="Survived",multiple="dodge")
sns.histplot(ax=axs[1,1], data=df,x="SibSp",  discrete=True,hue="Survived",multiple="dodge" )
sns.histplot(ax=axs[1,2], data=df,x="Parch",  discrete=True, hue="Survived",multiple="dodge" )
plt.show()

# Fare
sns.histplot(data=df,x="Fare", hue="Survived",multiple="dodge")
plt.show()
#Class
sns.countplot(x="Pclass",data=df, order=[1,2,3], hue="Survived")
plt.show()
plt.show()
#? Conclutions 

"""
- Survival rate of women is higher than men 
- survival rate of 1st class passengers is higher than other Pclass
- Children younger than 7 years survived more than died 
- People in the age range 48-51 survived mkore than died 
- Higher Fare have higher survival ratio (IMPORTANT: these are the outliers removed)
"""


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



#* DATA CLEANING 


#Compute Interquartile range to detect outliers        
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
"""plt.subplot(221)
df.Age.plot(kind="box", title = "Outliers")
plt.subplot(222)
df2.Age.plot(kind="box",  title = "No-outliers")
plt.subplot(223)
df.Fare.plot(kind="box",  title = "Outliers")
plt.subplot(224)
df2.Fare.plot(kind="box",  title = "No-outliers")
plt.show()"""


#print(df.describe())
print(df2.describe())

