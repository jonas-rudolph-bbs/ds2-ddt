import pandas as pd
import numpy as np
import math


class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self):
        """
        Generate a simple profile of the DataFrame, including:
          - Column data types
          - Number of missing values per column
          - Basic statistics for numeric columns
        """
        profile = {}
        return profile
    