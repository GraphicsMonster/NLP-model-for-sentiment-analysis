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


path1 = './Dataset/Reddit_Data.csv'
path2 = './Dataset/twitter_Data.csv'

df1 = load_data(path1)
df2 = load_data(path2)
print(df1.head())
print(df2.head())
