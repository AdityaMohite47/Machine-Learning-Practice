import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Columns : 

# age: age of primary beneficiary

# sex: insurance contractor gender, female, male

# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

# children: Number of children covered by health insurance / Number of dependents

# smoker: Smoking

# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

# charges: Individual medical costs billed by health insurance. (Target variable)


#-----------------------------------------------------------------------------------------------

# importing dataset , performing some analytics and corrections

df = pd.read_csv(r'C:\Machine-Learning-Practice\Machine-Learning-Practice\Datasets\insurance.csv')
# print(df.head())
# print(df.info())

cleaned_data = {
    'sex': {'male': 0, 'female': 1},
    'smoker': {'no': 0, 'yes': 1},
    'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
}

pd.set_option('future.no_silent_downcasting', True)
df_copy = df.copy()
df_copy.replace(cleaned_data , inplace=True)
df_copy = df_copy.infer_objects(copy=False)

# determing the correlation of other columns with the traget
# print(df_copy.corr()['charges'].sort_values())
# sns.heatmap(data=df_copy.corr() , annot=True)
# plt.show()

#-------------------------------------------------------------------------------------

# Making necessary imports 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.metrics import mean_absolute_error , mean_squared_error

# defining features and labels
poly = PolynomialFeatures(degree=2 , include_bias=False)
X = poly.fit_transform(df_copy.drop('charges' , axis=1))
y = df_copy['charges']

# creating train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# scaling data
scaler = StandardScaler()
scaled_X_train= scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Defining the model 

Model = ElasticNet(max_iter=5000)

params_grid = {
    'alpha':[x for x in range(1 ,  31)],
    'l1_ratio':[0.1, 0.2 , 0.3 , 0.4 , 0.5 , 0.6, 0.9, 1.0]
}

grid_search_model = GridSearchCV(estimator=Model , param_grid=params_grid , scoring='neg_root_mean_squared_error' , cv=125)
grid_search_model.fit(scaled_X_train , y_train)

# print(grid_search_model.best_params_)
# print(grid_search_model.best_score_)

y_pred = grid_search_model.predict(scaled_X_test)
print(mean_absolute_error(y_test , y_pred))
print(mean_squared_error(y_test , y_pred))
print(np.sqrt(mean_squared_error(y_test , y_pred)))