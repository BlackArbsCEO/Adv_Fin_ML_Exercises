import pandas as pd
import numpy as np

def cprint(df):
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_frame()
        except:
            raise ValueError('object cannot be coerced to df')

    print('-'*79)
    print('dataframe information')
    print('-'*79)
    print(df.tail(5))
    print('-'*50)
    print(df.info())
    print('-'*79)
    print()

get_range = lambda df, col: (df[col].min(), df[col].max())

