import pandas as pd
import numpy as np
import os

def load_data(path):
    """
    Loading data from path1 and path2
    """
    df = pd.read_csv(path)
    return df

    # Now for this model I am using csv files with only 2 fields text and a sentiment label
    # I am using the sentiment label as the target variable
