# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NITHYASRI M
RegisterNumber:  212224040226
*/
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()

print("data.info()")

df.info()
print("data.isnull().sum()")
df.isnull().sum()

print("data value counts")
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()

print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/9b89e2b1-6e34-45bd-9478-ce2cafa67803)
![image](https://github.com/user-attachments/assets/e67c7402-d35e-4f86-ab41-c699c07b373c)
![image](https://github.com/user-attachments/assets/b55ca2f3-0797-49ff-b0ae-79e65c7d1789)
![image](https://github.com/user-attachments/assets/720bcd2b-2ff3-45f5-8e7d-48ead9b0e7ec)
![image](https://github.com/user-attachments/assets/093d5c38-9924-4016-8134-864af4fd3b78)
![image](https://github.com/user-attachments/assets/40bec6f3-9d3f-4348-863c-1928fe9cbefe)
![image](https://github.com/user-attachments/assets/8d8e910f-3664-4e55-8b94-0e4fa5fce0d4)
![image](https://github.com/user-attachments/assets/bef7721d-65a9-4ee1-a458-58cb128d3787)
![image](https://github.com/user-attachments/assets/5cf0fd68-9e20-48bf-8bb4-d02f58fce005)
![image](https://github.com/user-attachments/assets/6cc4a51c-ab95-4497-ac27-2def194136c8)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
