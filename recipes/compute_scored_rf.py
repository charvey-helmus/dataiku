# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
custom_random_forest = dataiku.Folder("HT8V6csb").get_path()
test = dataiku.Dataset("test")
test_df = test.get_dataframe()

from sklearn.externals import joblib
clf = joblib.load(os.path.join(custom_random_forest, "clf.pkl"))
scores = clf.predict(test_df)
probas = clf.predict_proba(test_df)
preds = pd.DataFrame(scores, index=test_df.index).rename(columns={0: 'prediction'})
pred_probas = pd.DataFrame(probas, index=test_df.index).rename(columns={0: 'proba_False', 1: 'proba_True'})
all_preds = test_df.join(preds)
all_preds = all_preds.join(pred_probas)


# Write recipe outputs
scored_rf = dataiku.Dataset("scored_rf")
scored_rf.write_with_schema(all_preds)
