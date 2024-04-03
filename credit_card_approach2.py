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
cc_apps.info()
cc_apps.isna().sum()
cc_apps.head()

# Drop the features 11 and 13
cc_apps = cc_apps.drop([11, 13], axis=1)

cc_apps_nans_replaced = cc_apps.replace("?", np.NaN)
cc_apps_nans_replaced[1] = cc_apps_nans_replaced[1].astype(float)

cc_apps_imputed = cc_apps_nans_replaced.fillna(cc_apps_nans_replaced.mean())

for col in cc_apps_imputed.columns:
    if cc_apps_imputed[col].dtype == "Object":
        cc_apps_imputed = cc_apps_imputed[col].fillna(
            cc_apps_imputed[col].value_counts().index[0])

cc_apps_imputed_cat_dummies = pd.get_dummies(cc_apps_imputed, drop_first=True)
cc_apps_imputed_cat_dummies.columns

X = cc_apps_imputed_cat_dummies.iloc[:, :-1].values
y = cc_apps_imputed_cat_dummies.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train, y_train)
scaledX_train = scaler.transform(X_train)
scaledX_test = scaler.transform(X_test)

logreg =LogisticRegression()
logreg.fit(scaledX_train, y_train)
y_pred = logreg.predict(scaledX_test)
confusion_matrix(y_test, y_pred)


tol = [0.01, 0.001, 0.00001]
max_iter = [100, 150, 200]

param_grid = {'tol':tol, 'max_iter':max_iter}

grid_log_model = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid_log_model.fit(scaledX_train, y_train)
grid_log_model.best_estimator_
grid_log_model.best_params_
grid_log_model.best_score_

y_pred_grid = grid_log_model.predict(scaledX_test)
confusion_matrix(y_test, y_pred_grid)
