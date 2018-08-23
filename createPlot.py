import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as sm
import pylab as plt
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import FuncTickFormatter, LabelSet, Label, ColumnDataSource, TextAnnotation
from bokeh.models import HoverTool
from bokeh.io import show, push_notebook, output_notebook, output_file

plt.style.use('bmh')

HOME = str(Path.home())

PLOT_DIR = os.path.join(HOME, 'source/presentationEAVM/plots')

def RAD(deg):
    return deg * np.pi / 180.0


def CREATE_STATS_PLOTS(num_seg):
    df = data_lv.loc[data_lv.Segm == num_seg]
    result = sm.ols(formula="Voltage ~ WT", data=df).fit()
    fig, ax = plt.subplots(nrows=3, figsize=(3, 9))
    fig.tight_layout(pad=1)
    sns.pointplot(x='WT', y='Voltage', data=df, ax=ax[2])
    ax[2].set_title('Conf. intervals')
    sns.regplot(x='WT', y='Voltage', data=df, ax=ax[0])
    ax[0].set_title('Regression plot')
    sns.distplot(result.resid, ax=ax[1])
    ax[1].set_title('Residuals')
    plt.savefig(os.path.join(PLOT_DIR, '{}.png'.format(num_seg)))
    plt.close()


def GET_R2(num_seg):
    df = data_lv.loc[data_lv.Segm == num_seg]
    result = sm.ols(formula="Voltage ~ WT", data=df).fit()
    return result.rsquared


def GET_SLOPE(num_seg):
    df = data_lv.loc[data_lv.Segm == num_seg]
    result = sm.ols(formula="Voltage ~ WT", data=df).fit()
    return result.params.WT


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='.csv file with data')
    args = parser.parse_args()

    if args.f is None:
        raise ValueError('No path to data')

    data_lv = pd.read_csv(args.f, index_col=0)
    folder = args.f.split('/')[-1].split('.')[0]
    PLOT_DIR = os.path.join(PLOT_DIR, folder)
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    for n in tqdm(range(1, 18)):
        CREATE_STATS_PLOTS(n)
