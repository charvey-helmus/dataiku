# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Importing mlflow models in DSS
# 
# In this notebook we show through a simple example how to import a machine learning model trained *entirely out of DSS* into a SavedModel in a project's Flow. We use the [Catboost]() framework to perform a binary classification task on the [UCI Bank dataset]().

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import dataikuapi
import os

from dataikuapi.dss.ml import DSSPredictionMLTaskSettings

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Step 1: train your model outside of DSS
# 
# Using the archive data and source files provided along with this notebook, perform the following actions *outside of DSS*:
# * Create a virtual environment using Python >= 3.6 and install the packages listed in `requirement.txt`
# * Activate the newly-created virtual environment
# * Go to `src/` and train the model by running `python train_catboost.py`. The resulting model artifact will be stored in the `dist/` directory, its name should be of the form `catboost-uci-bank-xxxxxxxx-xxxxxx`.
# 
# > **WARNING**: Any pre-processing step applied to the training data **MUST** also be applied to the evaluation data.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Step 2: create the code env in DSS
# 
# In the *Administration > Code envs* section of DSS, crate a new code environment and add the packages listed in the archive's `requirement.txt` file (minus `pandas`), then build the code-env.
# 
# > **This notebook should be running using that code env ! **
# 
# Write down the name of that code env, you will need it to call `import_mlflow_version_from_path()`.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Step 3: get a handle on a SavedModel

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()

# Get or create SavedModel
sm_name = "catboost-uci-bank"
sm_id = None
for sm in project.list_saved_models():
    if sm_name != sm["name"]:
        continue
    else:
        sm_id = sm["id"]
        print("Found SavedModel {} with id {}".format(sm_name, sm_id))
        break
if sm_id:
    sm = project.get_saved_model(sm_id)
else:
    sm = project.create_mlflow_pyfunc_model(name=sm_name,
                                            prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.BINARY)
    sm_id = sm.id
    print("SavedModel not found, created new one with id {}".format(sm_id))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Step 4: import the evaluation dataset
# 
# Create a new Dataset in your DSS project by uploading `data/uci-bank-marketing/eval_data.csv`. Call this Dataset `eval_data`.
# 
# > **WARNING**: The evaluation Dataset **MUST** already be preprocessed using the exact same steps as in step 1 !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Step 5: Import mlflow model into a SavedModel version

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Change the following values to match your setup !
MLFLOW_DIST_DIR = "/Users/chrishelmus/Library/dss/design/managed_folders/ML_FLOW/cL2K5bqK"
CATBOOST_MODEL_DIR = "cat_boost_model_v2"

version_id = "v06" # Change this to iterate to a new version
model_dir = os.path.join(MLFLOW_DIST_DIR, CATBOOST_MODEL_DIR)

print(model_dir)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create version in SavedModel
for v in sm.list_versions():
    if v["id"] == version_id:
        raise Exception("SavedModel version already exists! Choose a new version name.")

sm_version = sm.import_mlflow_version_from_path(version_id=version_id,
                                                path=model_dir,
                                                code_env_name="ml_flow_py37")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Evaluate the version using the previously created Dataset
sm_version.set_core_metadata(target_column_name="y",
                             class_labels=["no", "yes"],
                             get_features_from_dataset="eval_data")
sm_version.evaluate("eval_data")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# If you go to the SavedModel's version screen, you should now be able to see properly all the "Performance" visualizations.