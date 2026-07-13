import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv("/content/AIML Dataset.csv")
data.head()
data.info()
data["isFraud"].value_counts()
data['isFlaggedFraud'].value_counts()
data.isnull().sum()
data.shape
data['type'].value_counts().plot(kind='bar',title='bar graph for type of payments')
fraud_by_type=data.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fraud_by_type.plot(kind='bar',title='fraud by which type')
data['amount'].describe().astype('float')
sns.boxplot(data[data['amount']<50000],x='isFraud',y='amount')
plt.title('amount vs isfraud under 50k')
data['balancedifforig']=data['oldbalanceOrg']-data['newbalanceOrig']
data['balancedestdiff']=data['newbalanceDest']-data['oldbalanceDest']
data['balancedifforig']<0
(data['balancedestdiff']<0).sum()
fraud_per_step=data[data['isFraud']==1]['step'].value_counts().sort_index()
plt.plot(fraud_per_step.index,fraud_per_step.values,label='fraud per step')
plt.show()
top_senders=data['nameOrig'].value_counts()
top_recivers=data['nameDest'].value_counts()
fraud_users=data[data['isFraud']==1]['nameOrig'].value_counts()
fraud_users.head()
fraud_types=data[data['type'].isin(['TRANSFER','CASH_OUT'])]
fraud_types['type'].value_counts()
sns.countplot(data=fraud_types,x='type',hue='isFraud')
correlation=data[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']].corr()
zero_after_transfer=data[
    (data['oldbalanceOrg']>0) &
    (data['newbalanceOrig']==0) &
    (data['type'].isin(['TRANSFER','CASH_OUT']))
]
data['isFraud'].value_counts()
data = data.dropna(subset=['isFraud'])
X=data.drop('isFraud',axis=1)
Y=data['isFraud']
data.drop(['nameOrig','nameDest','isFlaggedFraud',],axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0,stratify=Y)
categorical=['type']
numeric=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','balancedifforig','balancedestdiff']

preprocessor=ColumnTransformer(
   transformers=[
       ('num',StandardScaler(),numeric),
       ('cat',OneHotEncoder(drop='first'),categorical)
   ],                   
   remainder='drop'
)
# pipeline 
pipe=Pipeline(
    [
    ('pre',preprocessor),
    ('model',LogisticRegression(class_weight='balanced',max_iter=1000))
    ]
)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
# metrics and checking performance 
print(classification_report(y_pred,y_test))
confusion_matrix(y_pred,y_test)
pipe.score(x_test,y_test)




