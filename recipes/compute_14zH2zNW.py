# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu





# Write recipe outputs
s3_folder = dataiku.Folder("14zH2zNW")
s3_folder_info = s3_folder.get_info()
