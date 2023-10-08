# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM :

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### STEP 1 :

Import the required libraries.

### STEP 2 :

Upload and read the dataset.

### STEP 3 :

Check for any null values using the isnull() function.


### STEP 4 :

From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.


## Program :


Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
### Developed by : ABRIN NISHA A
### RegisterNumber : 212222230005  

```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:

### Initial data set :

![AA1](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/ec1e041a-7df4-4212-8b52-be4a5d79f208)

### Data info :

![AA2](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/9738c214-b82d-4761-b7f8-ce4a9ba6ae58)

### Optimization of null values :

![AA3](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/06b4db5b-a878-4367-a96a-a8ffc747c62c)

### Assignment of x value :

![AA4](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/18f45fd3-695f-4733-b7a2-f21f7169e462)

### Assignment of y value :

![AA5](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/29483f1b-ecaa-4870-bd8c-f4557b317760)

### Converting string literals to numerical values using label encoder :

![AA6](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/aa440e1b-9e5c-4d3e-ac8a-a35e0246e61e)


### Accuracy :

![AA7](https://github.com/Abrinnisha6/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118889454/b9846165-9a1c-45ff-8b85-7c99a541a386)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
