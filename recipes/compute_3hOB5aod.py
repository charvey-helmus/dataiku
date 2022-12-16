# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
# Read recipe inputs
flight_joined_prepared = dataiku.Dataset("flight_data")
df = flight_joined_prepared.get_dataframe()

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
import dataiku
import mlflow


project = dataiku.api_client().get_default_project()
managed_folder = project.get_managed_folder('3hOB5aod')

with project.setup_mlflow(managed_folder=managed_folder) as mlflow:
    mlflow.set_experiment("experiment_2")

    # activate Mflow autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="my_run"):
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("clf", RandomForestClassifier())])

        param_grid = {
            "clf__max_depth"        : [3],
            "clf__max_features"     : [1],
            "clf__min_samples_split": [2],
            "clf__min_samples_leaf" : [1],
            "clf__bootstrap"        : [False],
            "clf__criterion"        : ["gini"],
            "clf__n_estimators"     : [10]
        }

        gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc', cv=3)
        X = df[SCHEMA['features_num'] + SCHEMA['features_cat']]
        Y = df[SCHEMA['target']].values
        gs.fit(X, Y)
        clf = gs.best_estimator_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_project('MLFLOWO16N')
# Get or create saved models
if dataiku.get_custom_variables()["saved_model_id"] == "":
    saved_model = project.create_mlflow_pyfunc_model("mlflow_model", "BINARY_CLASSIFICATION")
    project.update_variables({"saved_model_id": saved_model.id})
else:
    saved_model = project.get_saved_model(dataiku.get_custom_variables()["saved_model_id"])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
mlflow_version = saved_model.import_mlflow_version_from_managed_folder(dataiku.get_custom_variables()["custom_model_version"], "3hOB5aod", 'experiment_2/my_run_jHL/artifacts/model/',code_env_name="mlflow_latest")
project.update_variables({"custom_model_version": int(dataiku.get_custom_variables()["custom_model_version"]) + 1})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
mlflow_version.set_core_metadata(SCHEMA['target'], class_labels=["False", "True"], get_features_from_dataset="flight_ground_truth_prepared")
mlflow_version.evaluate("flight_ground_truth_prepared")