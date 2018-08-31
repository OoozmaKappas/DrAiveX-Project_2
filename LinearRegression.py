import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

DataFeatures=pd.read_csv(r'F:\LetsCode\DrAIvX\Project 02\train.csv')#importing training data

#Filling up NaN values and also chaging sex From (MAle,Female) to (0,1)
DataFeatures["Fare"]=DataFeatures["Fare"].fillna(32.204)
DataFeatures["Age"]=DataFeatures["Age"].fillna(29.699118)
DataFeatures["Sex"]=[1 if items == 'female' else 0 for items in DataFeatures.Sex]

#creating the y axis data for fit in training
y_Labels=DataFeatures["Survived"]

#keeping on the important attributes and dropping the unnecessary details
x_Labels=DataFeatures.drop(["Name","Ticket","Cabin","Embarked","Survived"],axis=1)

#creating object of the Linear Regression class
Lin_reg = LinearRegression()
Lin_reg.fit(x_Labels,y_Labels)#fitting the data from the training values

#import train data
DataTestX=pd.read_csv(r'F:\LetsCode\DrAIvX\Project 02\test_X.csv')
DataTestY=pd.read_csv(r'F:\LetsCode\DrAIvX\Project 02\test_Y.csv')

#Modifying the test data by filling the NaN and changing Sex to binary
DataTestX["Sex"]=[1 if items == 'female' else 0 for items in DataTestX.Sex]
DataTestX["Fare"]=DataTestX["Fare"].fillna(32.204)
DataTestX["Age"]=DataTestX["Age"].fillna(29.699118)

#including only essential attributes
TestX=DataTestX.drop(["Name","Ticket","Cabin","Embarked"],axis=1)
TestY=DataTestY.drop(["PassengerId"],axis=1)

#Creating object of the Logistic regression class
Log_reg = LogisticRegression()
Log_reg.fit(x_Labels,y_Labels)

#printing the score of the respective type of regression 
print(Lin_reg.score(TestX,TestY))
print(Log_reg.score(TestX,TestY))
