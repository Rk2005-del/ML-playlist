import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import  make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import r2_score,accuracy_score,precision_score

data = pd.read_csv('Bengaluru_House_Data.csv')
data.head(5)
data.shape
data.info()
data.isnull()
for col in data.columns:
  print(data[col].value_counts())
#   print("*"*20)
data.isna().sum()
# REMOVING USELESS COLUMNS
data.drop(columns=['society','balcony'],inplace=True)
# MISSING VALUES
data['location'].value_counts()
data['location']=data['location'].fillna("Whitefield")
data['size'].value_counts()
data['size']=data['size'].fillna("2 BHK")
data['bath']=data['bath'].fillna(data['bath'].median())
data['bhk']=data['size'].str.split().str.get(0).astype(int)
# /HERE IN MANY ARE IN RANGE SO WE TAKE A AVERAGE VALUE AND FILL IT WITH IT
data['total_sqft'].unique()
def convertrange(x):
  temp=x.split('-')
  if len(temp)==2:
    return (float(temp[0])+float(temp[1]))/2
  try:
    return float(x)
  except:
    return None    
  
data['total_sqft']=data['total_sqft'].apply(convertrange)
# WE ARE TAKING PRICE PER SQUARE FEET 
data['price_per_squarearea']=data['price']*100000/data['total_sqft']

# THERE ARE MANY LOCATION WHOSE COUNT IS 1 OR LESS THAN 10 SO WE MAKE THEM OTHERS
data['location'].value_counts()
data['location']=data['location'].apply(lambda x:x.strip())
loc_count=data['location'].value_counts()
loc_count_less10=loc_count[loc_count<=10]
# loc_count_less10
data['location']=data['location'].apply(lambda x :'other' if x in loc_count_less10 else x)

data['price_per_squarearea'].describe()
(data['total_sqft']/data['bhk']).describe()
data = data[(data['total_sqft'] / data['bhk']) >= 300]
data.describe()
# REMOVING OUTLIERS IN SQFT
def remove_outlier_sqft(df):
  df_output=pd.DataFrame()
  for key,subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_squarearea)
    st=np.std(subdf.price_per_squarearea)
     
    gen_df=subdf[(subdf.price_per_squarearea>(m-st)) & (subdf.price_per_squarearea<=(m+st))]
    df_output=pd.concat([df_output,gen_df],ignore_index=True) 
  return df_output 
data=remove_outlier_sqft(data)
data.describe()
# REMOVING OUTLIER IN BHK
def remove_outlier_bhk(df):
  exclude_indices = np.array([], dtype=int)
  for loc,loc_df in df.groupby('location'):
    bhk_stats={}
    for bhk,bhk_df in loc_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean':np.mean(bhk_df.price_per_squarearea),
          'std':np.std(bhk_df.price_per_squarearea),
          'count':bhk_df.shape[0]
      }
    for bhk,bhk_df in loc_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_squarearea<(stats['mean'])].index.values)
  return df.drop(exclude_indices,axis='index')    
data=remove_outlier_bhk(data)
# BHK IN INTEGER
data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else None)
data=data.drop('size',axis=1)
# CLEANED DATA 
data.to_csv('cleaned_data.csv')
# TAKING X AND Y 
X=data.drop(columns=['price'])
Y=data['price']

X_train,X_text,Y_train,Y_text=train_test_split(X,Y,test_size=0.2,random_state=0)
# ONE HOT ENCODING ON LOCATION,AREA_TYPE AND AVAILABILITY
columns_trans=make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore',sparse_output=False),
     ['location','area_type','availability']
     ),
    remainder='passthrough'
    )
# SCALING THE VALUES
scaler=StandardScaler()
# lINEAR REGRESSION
lr=LinearRegression()
pipe=make_pipeline(columns_trans,scaler,lr)
pipe.fit(X_train,Y_train)
y_pred_lr=pipe.predict(X_text)
r2_score(y_pred_lr,Y_text)
# LASSO REGULARIZATION
r1=Lasso()
pipe=make_pipeline(columns_trans,scaler,r1)
pipe.fit(X_train,Y_train)
y_pred_lasso=pipe.predict(X_text)
r2_score(y_pred_lasso,Y_text)

# RIDGE REGURALIZATION
r2=Ridge()
pipe=make_pipeline(columns_trans,scaler,r2)
pipe.fit(X_train,Y_train)
y_pred_ridge=pipe.predict(X_text)
r2_score(y_pred_ridge,Y_text)