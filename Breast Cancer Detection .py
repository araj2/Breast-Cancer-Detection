import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline (if you are using jupyter notebook)

#Get the data
from sklearn. datasets import load_breast_cancer
cancer = load_breast_cancer()
# importing data from the builtin

cancer.keys()
#I am going to grab information and arayys out of this dictionary to set up our dataframe
print(cancaer['DESCR'])
cancer['feature_names']
df_feat = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])
df_feat.info()

cancer['target']
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])

df.head()

#Train Test Split

from sklearn.model_selection import train_test_split
model = svc()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
from sklearn.metrics import classifcation_report,confusion_matrix
print(confusion_matrix,(y_test,predictions))
print(classification_report(y_test,predictions))
#the above will predict that no tumors are in the 0 class, sci-kit learn wil warn you that
#precision and  fscore are ill defined and are being set to 0 for no predictied sample.
#It predicted that everything belonged to class 1

#our model needs to have the parameters adjusted and normalize the data since you are passing it into support vector machine
#to search for the parameters you can use a grid search
#grid search allows you to find the right parameters.

#Gridsearch

param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose =3)
grid.fit(X_train,y_train)
grid.best_params
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_Test,grid_predictions))
print(classifcation_report,(y_test,grid_predictions))
