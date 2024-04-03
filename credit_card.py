import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

cc_apps = pd.read_csv('C:/Users/patilni/OneDrive - Kantar/Desktop/Projects/Predicting Credit Card Approvals/credit+approval/crx.data', header=None)

cc_apps.shape
cc_apps.columns
cc_apps.describe()
cc_apps.isna().sum()
cc_apps.head()

# Drop the features 11 and 13
cc_apps = cc_apps.drop([11, 13], axis=1)

# Split into train and test sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

# Replace the '?'s with NaN in the train and test sets
cc_apps_train_nans_replaced = cc_apps_train.replace("?", np.NaN)
cc_apps_test_nans_replaced = cc_apps_test.replace("?", np.NaN)

# Impute the missing values with mean imputation
cc_apps_train_imputed = cc_apps_train_nans_replaced.fillna(cc_apps_train_nans_replaced.mean())
cc_apps_test_imputed = cc_apps_test_nans_replaced.fillna(cc_apps_test_nans_replaced.mean())

for col in cc_apps_train_imputed.columns:
    if cc_apps_train_imputed[col].dtype == 'Object':
        cc_apps_train_imputed = cc_apps_train_imputed.fillna(
            cc_apps_train_imputed[col].value_counts().index[0])
        cc_apps_test_imputed = cc_apps_test_imputed.fillna(
            cc_apps_test_imputed[col].value_counts().index[0])

cc_apps_train_cat_encoding = pd.get_dummies(cc_apps_train_imputed)
cc_apps_test_cat_encoding = pd.get_dummies(cc_apps_test_imputed)

cc_apps_test_cat_encoding = cc_apps_test_cat_encoding.reindex(columns=cc_apps_train_cat_encoding.columns, fill_value=0)

X_train, y_train = (
    cc_apps_train_cat_encoding.iloc[:, :-1].values,
    cc_apps_train_cat_encoding.iloc[:, [-1]].values,
    )

X_test, y_test = (
    cc_apps_test_cat_encoding.iloc[:, :-1].values,
    cc_apps_test_cat_encoding.iloc[:, [-1]],
    )

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)
y_pred = logreg.predict(rescaledX_test)
confusion_matrix(y_test, y_pred)

tol = [0.01, 0.001, 0.00001]
max_iter = [100, 150, 200]
param_grid = {'tol': tol, 'max_iter': max_iter}

grid_model = GridSearchCV(logreg, param_grid=param_grid,cv=5)
grid_model_result = grid_model.fit(rescaledX_train, y_train)

best_estimator = grid_model_result.best_estimator_
best_params = grid_model_result.best_params_
best_score = grid_model_result.best_score_

best_model = grid_model_result.best_estimator_
best_model.score(rescaledX_test, y_test)


