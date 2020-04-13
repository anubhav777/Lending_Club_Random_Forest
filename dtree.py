import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('loan_data.csv')
print(df.info())
print(df.head())
cd=df[df['credit.policy'] == 1]
zer=df[df['credit.policy'] == 0]
plt.hist([zer['fico'],cd['fico']],histtype='barstacked',bins=30, alpha=0.7)
plt.hist(df['fico'],histtype='barstacked')
one=df[df['not.fully.paid'] == 1]['fico']
print(one)
zero=df[df['not.fully.paid'] == 0]['fico']
plt.hist([one,zero],stacked=True,alpha=0.7,bins=30)
plt.figure(figsize=(10,5))
sns.countplot(x=df['purpose'],hue=df['not.fully.paid'])
sns.jointplot(df['fico'],df['int.rate'])
sns.lmplot(x='fico',y='int.rate',data=df,col='not.fully.paid',hue='credit.policy')
df=pd.read_csv('loan_data.csv')
po=pd.get_dummies(df['purpose'],drop_first=True)
df.drop(labels='purpose',axis=1,inplace=True)
final_data=pd.concat([df,po],axis=1)
X=final_data.drop(labels='not.fully.paid',axis=1).values
y=df['not.fully.paid']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
rf=RandomForestClassifier(n_estimators=300)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


