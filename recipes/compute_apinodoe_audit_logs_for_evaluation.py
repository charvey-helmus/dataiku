# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
input_dataset = dataiku.Dataset("apinode_audit_logs_current_project")


df = input_dataset.get_dataframe().dropna(axis=1, how='all')


# Write recipe outputs
testing = dataiku.Dataset("apinodoe_audit_logs_for_evaluation")
testing.write_with_schema(df)