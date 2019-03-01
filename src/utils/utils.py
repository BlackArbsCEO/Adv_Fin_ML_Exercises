from pathlib import PurePath, Path
import sys
import time
import os
import json
os.environ['THEANO_FLAGS'] = 'device=cpu'
import pymc3 as pm
import pandas as pd
import numpy as np
import dask 
import dask.dataframe
import decimal
import logzero
from logzero import logger
import matplotlib.pyplot as plt
import seaborn as sns
from src.CONSTANTS import *
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)
#=============================================================================
## setup logger

def setup_system_logger(out_log_fp, pdir, logger):
    """fn: setup logger for various package modules

    Params
    ------
    out_log_fp: str
        log file fp name doesn't include extension fn will add it
    logger: logzero logger object

    Returns
    -------
    logger: logzero logger instance
    """
    now = pd.to_datetime('now', utc=True)
    file_ = out_log_fp+f'_{now.date()}.log'
    logfile = Path(pdir/'logs'/file_).as_posix()
    check_path(logfile)
    formatter = logzero.LogFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logzero.setup_default_logger(logfile=logfile, formatter=formatter)
    return logger

#=============================================================================
# general utils

def get_relative_project_dir(project_repo_name=None, partial=True):
    """helper fn to get local project directory"""
    current_working_directory = Path.cwd()
    cwd_parts = current_working_directory.parts
    if partial:
        while project_repo_name not in cwd_parts[-1]:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    else:
        while cwd_parts[-1] != project_repo_name:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    return current_working_directory

def check_path(fp):
    """fn: to create file directory if it doesn't exist"""
    if not Path(fp).exists():

        if len(Path(fp).suffix) > 0: # check if file
            Path(fp).parent.mkdir(exist_ok=True, parents=True)

        else: # or directory
            Path(fp).mkdir(exist_ok=True, parents=True)

def cprint(df):
    if not isinstance(df, (pd.DataFrame, dask.dataframe.DataFrame)):
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
#=============================================================================
# system utils
def decimal_round(val, prec=1e-4):
    """wrapper for rounding according to precision
    """
    DD = decimal.Decimal
    val_ = DD(val).quantize(DD(f'{prec}'), rounding=decimal.ROUND_DOWN)
    return float(val_)

#=============================================================================
# fn: code adapted from https://github.com/jonsedar/pymc3_vs_pystan/blob/master/convenience_functions.py
def custom_describe(df, nidx=3, nfeats=20):
    ''' Concat transposed topN rows, numerical desc & dtypes '''

    print(df.shape)
    nrows = df.shape[0]
    
    rndidx = np.random.randint(0,len(df),nidx)
    dfdesc = df.describe().T

    for col in ['mean','std']:
        dfdesc[col] = dfdesc[col].apply(lambda x: np.round(x,2))
 
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), axis=1, join='outer')
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)
    
    # add count nonNAN, min, max for string cols
    nan_sum = df.isnull().sum()
    dfout['count'] = nrows - nan_sum
    dfout['min'] = df.min().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['nunique'] = df.apply(pd.Series.nunique)
    dfout['nan_count'] = nan_sum
    dfout['pct_nan'] = nan_sum / nrows
    
    return dfout.iloc[:nfeats, :]


def plot_tsne(dftsne, ft_num, ft_endog='is_vw'):
    ''' Convenience fn: scatterplot t-sne rep with cat or cont color'''

    pal = 'cubehelix'
    leg = True

    if ft_endog in ft_num:
        pal = 'BuPu'
        leg = False

    g = sns.lmplot('x', 'y', dftsne.sort(ft_endog), hue=ft_endog
           ,palette=pal, fit_reg=False, size=7, legend=leg
           ,scatter_kws={'alpha':0.7,'s':100, 'edgecolor':'w', 'lw':0.4})
    _ = g.axes.flat[0].set_title('t-SNE rep colored by {}'.format(ft_endog))


def trace_median(x):
    return pd.Series(np.median(x,0), name='median')

def plot_traces(trcs, retain=2500, varnames=None):
    ''' Convenience fn: plot traces with overlaid means and values '''
    df_smry = pm.summary(trcs[-retain:], varnames=varnames)

    if varnames: nrows = len(varnames)
    else: nrows = len(trcs.varnames)
    
    plt.style.use('seaborn-dark-palette')
    plt.rcParams['font.family'] = 'DejaVu Sans Mono'
    line_cols = ['mean','hpd_2.5','hpd_97.5']
    ax = pm.traceplot(trcs[-retain:], varnames=varnames, figsize=(12, nrows*1.5), 
                      lines={k: v[line_cols[0]] for k,v in df_smry.iterrows()})

    for i,var in enumerate(df_smry.index):
        ax[i,0].axvline(df_smry.loc[var,line_cols[1]],color=red)
        ax[i,0].axvline(df_smry.loc[var,line_cols[2]], color=blue)

    for i, mn in enumerate(df_smry['mean']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='bottom', fontsize='large', color='#AA0022')
        except: 
            pass
        
    for i, mn in enumerate(df_smry['hpd_2.5']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,15), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='top', fontsize='medium', color='#AA0022')
        except: 
            pass
        
    for i, mn in enumerate(df_smry['hpd_97.5']):
        try:
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,15), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='top', fontsize='medium', color=blue)#'#AA0022')
        except: 
            pass  