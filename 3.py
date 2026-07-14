import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,r2_score,precision_score, mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

data=pd.read_csv('/content/demand_forecasting.csv')
data.head()
# EDA
data['Epidemic'].value_counts()
data.dtypes
data.info()
data.isnull().sum()
print(data[data.duplicated(keep=False)])
data.describe().T
data.describe(include='object').T
data['Date'] = pd.to_datetime(data['Date'])
data['year']=data['Date'].dt.year
data['month']=data['Date'].dt.month
data['day']=data['Date'].dt.day
data['weekday']=data['Date'].dt.day_name
data['discounted_price']=data['Price']*(1-data['Discount']/100)
data['sell_through_Rate']=data['Units Sold']/data['Inventory Level']
data['sell_through_Rate'].describe()
data.groupby('Category')["Demand"].agg(['mean','sum','std']).sort_values(by='sum',ascending=False).reset_index()
data.groupby(['Region','Seasonality'])['Demand'].mean()
data.groupby('Promotion')['Demand'].mean()
pd.pivot_table(data,values="Demand",index='month',columns='Category',aggfunc='mean')
sns.histplot(data['Demand'],bins=200,kde=True)
sns.scatterplot(data,x='Inventory Level',y='Units Sold')
sns.boxplot(data,x='Category',y='Demand')
data.groupby('Promotion')['Demand'].mean()
monthly_demand=data.groupby('month')['Demand'].mean()
monthly_demand.plot(kind='bar')
daily_demand=data.groupby('Date')['Demand'].sum()
daily_demand.plot()
plt.title('Demand over time')
plt.xlabel('date')
plt.ylabel('Demand')
plt.show()
sns.barplot(data,x='Promotion',y='Demand')
sns.scatterplot(data,x='discounted_price',y='Demand')
data.groupby('Seasonality')['Demand'].mean().plot(kind='bar',title='demand by show')
data.groupby('Epidemic')['Demand'].mean().plot(kind='bar',title='Epidemic impact on Demand')

# CLEANED DATA FILE 
data.to_csv('cleaned_demand_forecasting_data.csv')

data_m=pd.read_csv('/content/cleaned_demand_forecasting_data.csv')
features=[
    'Price',
    'Discount',
    'Inventory Level',
    'Promotion',
    'Competitor Pricing',
    'Category',
    'Seasonality'

]
target='Demand'
X=data[features].copy()
Y=data[target]
categorical_cols=X.select_dtypes(include='object').columns
numeric=['Price','Discount','Inventory Level','Promotion','Competitor Pricing']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# PIPELINING
preprocessor=ColumnTransformer(
    transformers=[
    ('clf',OneHotEncoder(),categorical_cols),
    ('num',MinMaxScaler(),numeric)
    ],
    remainder='passthrough'                           
)
pipe=Pipeline([
    ('pre',preprocessor),
    ('xgb',XGBRegressor(objective='reg:squarederror',n_jobs=-1)) 
])

params={
    'xgb__estimators':[200,300,500],
    'xgb__max_depth':[3,4,6,8],
    'xgb__learning_rate':[0.01,0.05,0.1],
    'xgb__subsample':[0.7,0.8,1.0],
    'xgb__colsample_bytree':[0.7,0.8,1.0],
    'xgb__min_child_weight':[1,3,5]
}
grid=GridSearchCV(
       estimator=pipe,      # Model to tune
        param_grid=params,# Hyperparameters to test
        cv=5,                 # 5-fold cross-validation
        scoring='neg_mean_absolute_error',# Evaluation metric
        n_jobs=-1,            # Use all CPU cores
        verbose=2             # Show progress
)

grid.fit(x_train,y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae)

# FEATURE IMPORTANCE

xgb_model = best_model.named_steps['xgb']
importance = xgb_model.feature_importances_

print(importance)
feature_names = best_model.named_steps['pre'].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print(importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])

plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()