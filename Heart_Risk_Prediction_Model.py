
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

# Found an Imbalance in classes thus procceding to resampling and testing again 

from imblearn.combine import SMOTETomek # UnderSampling the majority class

re_sampler = SMOTETomek(random_state=21 , sampling_strategy='minority')
re_X , re_y = re_sampler.fit_resample(X=X , y=y)

from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Suppressing Warnings
import warnings
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

X_train, X_test, y_train, y_test = train_test_split(re_X, re_y, test_size=0.4, random_state=21) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Base_Model = OneVsRestClassifier(estimator=LogisticRegression(solver='saga' , max_iter=5000))

params_grid = {
    'estimator__penalty' : ['l1' , 'l2' , 'elasticnet'],
    'estimator__C': [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9], 
    'estimator__l1_ratio': [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9]
}

Grid_Search_Model = GridSearchCV(estimator=Base_Model , param_grid=params_grid , cv=10)
Grid_Search_Model.fit(X_train , y_train)

print(Grid_Search_Model.best_params_) #{'estimator__C': 0.1, 'estimator__l1_ratio': 0.1, 'estimator__penalty': 'l1'}


# -------------------------------------------------------------------------------------------------------------------------------------


# Testing
from sklearn.metrics import confusion_matrix  ,classification_report

y_pred = Grid_Search_Model.predict(X_test)

print("Confusion Matix : \n" , confusion_matrix(y_test , y_pred))
# [[966 520]
#  [410 956]]

print("Classification Report : \n" , classification_report(y_test , y_pred))
#                precision    recall  f1-score   support

#            0       0.70      0.65      0.68      1486
#            1       0.65      0.70      0.67      1366

#     accuracy                           0.67      2852
#    macro avg       0.67      0.67      0.67      2852
# weighted avg       0.68      0.67      0.67      2852


# Above are the best scores I stretched with resampling the data.

