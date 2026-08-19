import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
housing=pd.read_csv("housing.csv")

# 2. Create a stratified test set
housing["income_cat"]=pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set= housing.loc[train_index].drop("income_cat", axis=1)
    start_test_set= housing.loc[test_index].drop("income_cat", axis=1)

#We'll only work on the copy of training data
housing=start_train_set.copy()

# 3. Seperate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. List numerical & categorical columns
num_attributes= housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes=["ocean_proximity"]

# 5. Pipelines- 
 
# i) Numerical columns
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#ii) Categorical columns
cat_pipeline=Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#iii) Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing) 

# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)

# 7. Train the model 

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
# lin_rmse= root_mean_squared_error(housing_labels, lin_preds)
# print(f"Linear Regression RMSE: {lin_rmse}")

#Decision Tree Regression
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds=dec_reg.predict(housing_prepared)
# dec_rmse= root_mean_squared_error(housing_labels, dec_preds)
# print(f"Decison Tree regression RMSE: {dec_rmse}")

#Random Forest Regressor
random_forest_reg=RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds=random_forest_reg.predict(housing_prepared)
# random_forest_rmse= root_mean_squared_error(housing_labels, random_forest_preds)
# print(f"Random Forest regression RMSE: {random_forest_rmse}")


# Evaluate Decision Tree with cross-validation
dec_rmses = -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.
print("Decision Tree CV RMSEs:", dec_rmses)
print("\nCross-Validation Performance (Decision Tree):")
print(pd.Series(dec_rmses).describe())

# Evaluate Linear Regression with cross-validation
lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("Linear Regression CV RMSEs:", lin_rmses)
print("\nCross-Validation Performance (Linear Regression):")
print(pd.Series(lin_rmses).describe())

# Evaluate Random Forest with cross-validation
random_forest_rmses = -cross_val_score(
    random_forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("Random Forest CV RMSEs:", random_forest_rmses)
print("\nCross-Validation Performance (Random Forest):")
print(pd.Series(random_forest_rmses).describe())