# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd, numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# load 'train' dataset as a Pandas dataframe
df = dataiku.Dataset("flight_data").get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-----------------------------------------------------------------
# Dataset Settings
#-----------------------------------------------------------------

# Select a subset of features to use for training
SCHEMA = {
    'target': 'Late',
    'features_num': ['dep_month', 'dep_woy', 'dep_hour','Distance','Late_avg'],
    'features_cat': ['UniqueCarrier', 'Origin','Dest']
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-----------------------------------------------------------------
# Preprocessing on Training Set
#-----------------------------------------------------------------

# Numerical variables
df_num = df[SCHEMA['features_num']]

trf_num = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sts', StandardScaler()),
])

# Categorical variables
df_cat = df[SCHEMA['features_cat']]

trf_cat = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", trf_num, SCHEMA['features_num']),
        ("cat", trf_cat, SCHEMA['features_cat'])
    ]
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-------------------------------------------------------------------------
# TRAINING
#-------------------------------------------------------------------------
##### TO-DO: add experiment tracking code here
##### but watch out for lineage (don't use the deploy button from the xperiment tracking UI)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("clf", RandomForestClassifier())]
)

param_grid = {
    "clf__max_depth"        : [3, None],
    "clf__max_features"     : [1, 3],
    "clf__min_samples_split": [2],
    "clf__min_samples_leaf" : [1],
    "clf__bootstrap"        : [True, False],
    "clf__criterion"        : ["gini", "entropy"],
    "clf__n_estimators"     : [10]
}

gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc', cv=3)
X = df[SCHEMA['features_num'] + SCHEMA['features_cat']]
Y = df[SCHEMA['target']].values
gs.fit(X, Y)
clf = gs.best_estimator_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# #-----------------------------------------------------------------
# # Score Test Set
# #-----------------------------------------------------------------

# # load 'test' dataset as a Pandas dataframe
# df_test = dataiku.Dataset("evaluation_data").get_dataframe()

# # Actually score the new records
# scores = clf.predict_proba(df_test)

# # Reshape
# preds = pd.DataFrame(scores, index=df_test.index).rename(columns={0: 'proba_False', 1: 'proba_True'})
# all_preds = df_test.join(preds)

# # Sample of the test dataset with predicted probabilities
# all_preds.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Compute AUC results
# auc = roc_auc_score(all_preds['high_value'].astype(bool).values, all_preds['proba_True'].values)
# auc

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
mlflow_models = dataiku.Folder("3hOB5aod").get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import mlflow

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#Save models as mlflow models to a folder
from datetime import datetime
import os
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = mlflow_models + "/custom-random-forest-{}".format(ts)
mlflow.sklearn.save_model(clf, model_dir)
print("Model saved at {} !".format(os.path.abspath(model_dir)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataikuapi

client = dataiku.api_client()
project = client.get_default_project()

# Get or create saved models
if dataiku.get_custom_variables()["saved_model_id"] == "":
    saved_model = project.create_mlflow_pyfunc_model("mlflow_model", "BINARY_CLASSIFICATION")
    project.update_variables({"saved_model_id": saved_model.id})
else:
    saved_model = project.get_saved_model(dataiku.get_custom_variables()["saved_model_id"])

mlflow_version = saved_model.import_mlflow_version_from_path(dataiku.get_custom_variables()["custom_model_version"], model_dir, code_env_name="ml_flow_py37")
project.update_variables({"custom_model_version": int(dataiku.get_custom_variables()["custom_model_version"]) + 1})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
mlflow_version.set_core_metadata(SCHEMA['target'], class_labels=["false", "true"], get_features_from_dataset="flight_ground_truth")
mlflow_version.evaluate("flight_ground_truth")