# The dataset includes 1500 rows and 7 columns:

# is_genuine: boolean
# diagonal: float
# height_left: float
# height_right: float
# margin_low: float
# margin_upper: float
# length: float
# Idea of projects with this dataset:

# Predicting the missing values with a linear regression or a KNN imputer
# Comparing classification such as logistic regression or KNN with an unsupervised model such as K-Means to predict the authenticity of the bills
# Trying to do a PCA or a Kernel Transform to create a clearer separation between the Genuine and Fake Bills.

#--------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from svm_margin_plot import plot_svm_boundary

data = pd.read_csv('./Datasets/fake_bills.csv' , sep=';')
# print(data.sample(10))

data['is_genuine'] = data['is_genuine'].map({True : 0 , False: 1})


# print(data.corr()['is_genuine'].sort_values()) # length feature is not the helping feature here since it has 
                                                # '-0.849285' correlation with traget label

data = data.drop('length' , axis=1)

# checking for null values 
# print(data.isnull().sum()) # 37 null values found in 'margin_low' feature\)

# Mean Imputing for null values
data['margin_low'] = data['margin_low'].fillna(data['margin_low'].mean())

X = data.drop('is_genuine' , axis=1)
y = data['is_genuine']

# print(y.value_counts()) # out of 1500 bills only 500 are fake.
# y.value_counts().plot.pie() # pandas in-built function to visualize a pie-chart
# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Making model (KNN)

# from sklearn.model_selection import train_test_split , GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.4)

# pipe = Pipeline([ ('scaler' , StandardScaler()) , ('model' , KNeighborsClassifier())])

# params_grid = {
#     'model__n_neighbors':[x for x in range(2 , 31)],
# }

# GridSearch_model = GridSearchCV(estimator=pipe , param_grid=params_grid , cv=10)
# GridSearch_model.fit(X_train , y_train)

# # print(GridSearch_model.best_params_) # {'model__n_neighbors': 3}

#---------------------------------------------------------------------------------------------------------------------------------------------------
# Making model (SVC)

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.4)

pipe = Pipeline([ ('scaler' , StandardScaler()) , ('model' , SVC())])

params_grid = {
    'model__C':[0.1 , 0.2 , 0.3 , 0.4 , 0.5],
    'model__kernel':['linear' , 'rbf' , 'poly'],
    'model__degree':[x for x in range(1 , 20)],
    'model__gamma':[0.1 , 0.2 , 0.3 , 0.4 , 0.5]
}

GridSearch_model = GridSearchCV(estimator=pipe , param_grid=params_grid , cv=10)
GridSearch_model.fit(X_train , y_train)

# print(GridSearch_model.best_params_) # {'model__C': 0.4, 'model__degree': 1, 'model__gamma': 0.2, 'model__kernel': 'rbf'}

#---------------------------------------------------------------------------------------------------------------------------------------------------

# Testing
from sklearn.metrics import confusion_matrix , classification_report
# # for KNN

# y_pred = GridSearch_model.predict(X_test)


# print("Confusion Matrix : \n" , confusion_matrix(y_test , y_pred))
# # Confusion Matrix : 
# #  [[379  13]
# #  [ 19 189]]

# print("Classification Report : \n" , classification_report(y_test , y_pred))
# # Classification Report : 
# #                precision    recall  f1-score   support

# #            0       0.95      0.97      0.96       392
# #            1       0.94      0.91      0.92       208

# #     accuracy                           0.95       600
# #    macro avg       0.94      0.94      0.94       600
# # weighted avg       0.95      0.95      0.95       600

# for Support Vector Classifier
y_pred = GridSearch_model.predict(X_test)


print("Confusion Matrix : \n" , confusion_matrix(y_test , y_pred))
# Confusion Matrix : 
#  [[385   7]
#  [ 15 193]]

print("Classification Report : \n" , classification_report(y_test , y_pred))
# Classification Report : 
#                precision    recall  f1-score   support

#            0       0.96      0.98      0.97       392
#            1       0.96      0.93      0.95       208

#     accuracy                           0.96       600
#    macro avg       0.96      0.96      0.96       600
# weighted avg       0.96      0.96      0.96       600
