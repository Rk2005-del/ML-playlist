import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


data=pd.read_csv('/content/email.csv')
data
data['Category'].unique()
data['Category'].value_counts()
data=data.iloc[:-1]
data.describe()
data.info()
data['Category'].value_counts(normalize=True)*100
# EDA
data['Category'].value_counts().plot(kind='pie',autopct='%1.1f%%')
data['Message_length']=data['Message'].apply(len)
data['Message_length'].describe()
sns.histplot(data['Message_length'],bins=20)
sns.boxplot(
    data,
    x='Category',
    y='Message_length'
)
data.groupby('Category')['Message_length'].mean()
data['words']=data['Message'].apply(lambda x :len(x.split()))
sns.histplot(
    data,
    x='words',
    hue='Category',
    bins=40
)
data.groupby('Category')['words'].mean()
data.nsmallest(10,'Message_length')
data.nlargest(10,'Message_length')
data['Category']=data['Category'].map({
    'ham':1,
     'spam':0
})
warnings.filterwarnings('ignore')

# train_test_series
X=data['Message']
Y=data['Category']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y)

pipe=Pipeline([
    # stop_words='english': Removes common words like "the", "is", "and".
# max_features=5000: Keeps only the 5,000 most important words.
# ngram_range=(1,2): Uses both single words (unigrams) and pairs of words (bigrams), e.g., "free" and "free prize".
# lowercase=True: Converts all text to lowercase before processing.
    ('tfidf',TfidfVectorizer( stop_words='english',max_features=5000, ngram_range=(1,2),lowercase=True )),
    ('model', MultinomialNB())
])

params=[
    
          # Naive Bayes
    {
        'model': [MultinomialNB()],
        'model__alpha': [0.1, 0.5, 1.0]
    },

    # Logistic Regression
    {
        'model': [LogisticRegression(max_iter=1000)],
        'model__C': [0.1, 1, 10]
    },

    # SVM
    {
        'model': [SVC()],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
          
      #Randomforestclassifier
   {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },

      # Decision tree
          {
         'model': [DecisionTreeClassifier(random_state=42)],
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
          },

      #Gradient Boosting
        {
        'model': [GradientBoostingClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5]
            
        }  
    
]


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(x_train, y_train)


Best_Model= grid.best_estimator_
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
y_pred=Best_Model.predict(x_test)
print(accuracy_score(y_pred,y_test))
print(f1_score(y_pred,y_test))

try:
  auc=roc_auc_score(y_test,Best_Model.decision_function(x_test))
except:
  try:
    auc=roc_auc_score(
        y_test,
        Best_Model.predict_proba(x_test)[:,1]

    )
  except:
      auc=np.nan

# Let try an example checking model working or not 

emails=[
    'Congratulation! you won a free prize. Click here now',
    'hey are we still meeting at 10 am tommmorow?',
    'Urgent ! claim your $1000 walmart gift card '
    
    ]
predictions=Best_Model.predict(emails)
for email,predictions in zip(emails,predictions):
  label='spam' if predictions==0 else 'ham'
  print(label,':','email')


with open('spam_classifier.pkl','wb') as file:
  pickle.dump(Best_Model,file)        

