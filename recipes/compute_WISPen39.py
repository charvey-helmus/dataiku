# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
my_pkl_model = dataiku.Folder("3hOB5aod")
my_pkl_model_info = my_pkl_model.get_info()

sm.import_mlflow_version_from_managed_folder( version_id="S3_ver", managed_folder=XP_TRACKING_FOLDER_ID, path="catboost20220516-151555")



# Write recipe outputs
output_hold = dataiku.Folder("WISPen39")
output_hold_info = output_hold.get_info()
