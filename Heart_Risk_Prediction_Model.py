
# World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in 
# the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases
# can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends 
# to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.

# Source
# The dataset is publically available on the Kaggle website, and it is from an ongoing cardiovascular study on residents of the town
#  of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).
# The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes. 


# Demographic [ Each attribute is a potential risk factor. There are both demographic, behavioral and medical risk factors. ] :

#   • Sex: male or female(Nominal).

#   • Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
#   Behavioral.

#   • Current Smoker: whether or not the patient is a current smoker (Nominal)

#   • Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can  
#       have any number of cigarettes, even half a cigarette.) Medical( history)

#   • BP Meds: whether or not the patient was on blood pressure medication (Nominal)

#   • Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)

#   • Prevalent Hyp: whether or not the patient was hypertensive (Nominal)

#   • Diabetes: whether or not the patient had diabetes (Nominal) , Medical(current)

#   • Tot Chol: total cholesterol level (Continuous)

#   • Sys BP: systolic blood pressure (Continuous)

#   • Dia BP: diastolic blood pressure (Continuous)

#   • BMI: Body Mass Index (Continuous)

#   • Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet 
#              are considered continuous because of large number of possible values.)

#   • Glucose: glucose level (Continuous)

#   Predict variable (desired target)
#   • 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)


# ------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# importing dataset , performing some analytics

data = pd.read_csv('./Datasets/framingham.csv')
# print(data.head())

# Determining the age group for CHD for both genders...
# sns.boxplot(x='TenYearCHD' , y='age' ,  data=data , hue='male')

# Determining the correlation of all features to each other...
# print(data.corr()['TenYearCHD'].sort_values())
# sns.heatmap(data=data.corr() , annot=True)
# plt.show()

# 'education' feature having no impact on CHD so dropping it , Correlation value : -0.054059
data = data.drop('education' , axis=1)

# checking for Null values
# print(data.isna().sum()) # 540 missing values found

# filling and dropping some missing values
data = data.dropna(subset=['heartRate']) # Only one missing value so dropping doesn't cost much

# Impute missing values for multiple columns
data['glucose'] = data['glucose'].fillna(data['glucose'].mean())  # Mean imputation for glucose
data['BPMeds'] = data['BPMeds'].fillna(data['BPMeds'].mode()[0])  # Mode imputation for BPMeds
data['totChol'] = data['totChol'].fillna(data['totChol'].median())  # Median imputation for totChol
data['cigsPerDay'] = data['cigsPerDay'].fillna(data['cigsPerDay'].median())  # Median imputation for cigsPerDay
data['BMI'] = data['BMI'].fillna(data['BMI'].mean())  # Mean imputation for BMI

# ----------------------------------------------------------------------------------------------------------------------------

# Designing data and Model

X = data.drop('TenYearCHD' , axis=1)
y = data['TenYearCHD']

# print(y.value_counts()) # Imbalance of classes Detected
# y.value_counts().plot.pie() # pandas in-built function to visualize a pie-chart
# plt.show()

from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression

# Suppressing Warnings
# import warnings
# warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Base_Model = OneVsRestClassifier(estimator=LogisticRegression(solver='saga' , max_iter=5000))

# params_grid = {
#     'estimator__penalty' : ['l1' , 'l2' , 'elasticnet'],
#     'estimator__C': [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9], 
#     'estimator__l1_ratio': [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9]
# }

# Grid_Search_Model = GridSearchCV(estimator=Base_Model , param_grid=params_grid , cv=10)
# Grid_Search_Model.fit(X_train , y_train)

# print(Grid_Search_Model.best_params_) #{'estimator__C': 0.1, 'estimator__l1_ratio': np.float64(0.0), 'estimator__penalty': 'l1'}


# ----------------------------------------------------------------------------------------------------------------------------

# Testing
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix , f1_score , classification_report

# y_pred = Grid_Search_Model.predict(X_test)

# print("Accuracy Score : " , accuracy_score(y_test , y_pred))
# print("Precision Score : " , precision_score(y_test , y_pred))
# print("Recall Score : " , recall_score(y_test , y_pred))
# print("f1 Score : " , f1_score(y_test , y_pred))
# print("Confusion Matix : \n" , confusion_matrix(y_test , y_pred))


#---------------------------------------------------------------------------------------------------------------------------------------------------

# Found an Imbalance in classes thus procceding to resampling and testing again 

# from imblearn.combine import SMOTETomek # Oversampling the majority class

# re_sampler = SMOTETomek(random_state=42)
# X_train , y_train = re_sampler.fit_resample(X_train , y_train)
# Grid_Search_Model.fit(X_train , y_train)
# print(Grid_Search_Model.best_params_)

# y_pred = Grid_Search_Model.predict(X_test)

# print("Accuracy Score : " , accuracy_score(y_test , y_pred))
# print("Precision Score : " , precision_score(y_test , y_pred))
# print("Recall Score : " , recall_score(y_test , y_pred))
# print("f1 Score : " , f1_score(y_test , y_pred))
# print("Confusion Matix : \n" , confusion_matrix(y_test , y_pred))

# Logistic Regression model didn't fit the problem statement well , looking for further improvements...

# Current Scores :  
# {'estimator__C': 0.5, 'estimator__l1_ratio': 0.6, 'estimator__penalty': 'elasticnet'}
# Accuracy Score :  0.659433962264151
# Precision Score :  0.2669902912621359
# Recall Score :  0.650887573964497
# f1 Score :  0.37865748709122204
# Confusion Matix :
#  [[589 302]
#  [ 59 110]]

# ----------------------------------------------------------------------------------------------------------------------------

# Solving the same with KNN

from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek

pipe = Pipeline(steps=[ ('scaler' , MinMaxScaler() ) , ('smote' , SMOTETomek())  , ('model' , KNeighborsClassifier(weights='distance'))])

params_grid = {
    'model__n_neighbors':[x for x in range(1 , 31)],
    'model__metric':['euclidean' , 'manhattan' , 'minkowski']
}

GridSearchModel_KNN = GridSearchCV(estimator=pipe , param_grid=params_grid , cv=10)
GridSearchModel_KNN.fit(X_train , y_train)

print(GridSearchModel_KNN.best_params_)

y_pred = GridSearchModel_KNN.predict(X_test)
print(classification_report(y_test , y_pred))
print(confusion_matrix(y_test , y_pred))

# KNN Classifier performs decent but not because of high frequency of datapoints in class "0" is far greater than class "1" creating imbalance.

# To be continued to develop a model that would solve the problem