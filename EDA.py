import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import os 

df = pd.read_csv("train.csv") #Reads data
describe = df.describe() #Summary of data
info = df.info(); 
nans = df.isna().sum() #Summary of NaN values in data
df =df.drop("Cabin",axis=1) # Cabin column is dropped because it has a lot of NaNs
df.interpolate(inplace=True) # perform linear interpolation to fill the NaNs
df.Embarked.fillna(df.Embarked.mode()[0], inplace=True) # replace NaNs in Embarked column with its mode

