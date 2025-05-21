# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Keziah.F
RegisterNumber:212223040094
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

##DATA HEAD

![image](https://github.com/user-attachments/assets/4852ffb6-2aec-48a8-ab2e-6754bbbb261d)

##DATSET INFO

![image](https://github.com/user-attachments/assets/72e213d4-1c5e-43d6-b4bc-acf0b79d32ec)

##NULL DATASET

![image](https://github.com/user-attachments/assets/6f001f9e-ef45-47de-9490-d81832bfe523)

##VALUES COUNT IN LEFT COLUMN

![image](https://github.com/user-attachments/assets/4c6f10ff-d2ea-43f2-8545-bb4e6b133d24)

##DATSET TRANSFORMED HEAD

![image](https://github.com/user-attachments/assets/eb02a391-4f34-4084-b17f-6e8bd05927db)

##X HEAD

![image](https://github.com/user-attachments/assets/e99fb5d0-24c7-49e1-a91d-8d2383c256aa)

##ACCURACY

![image](https://github.com/user-attachments/assets/b0566109-6cd8-45f6-aaee-47b3345d9d18)

##DATA PREDICTION

![image](https://github.com/user-attachments/assets/ecdf6efd-e6d4-42a1-9260-ade638581436)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
