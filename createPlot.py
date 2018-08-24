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


def CREATE_STATS_PLOTS(num_seg, out_dir):
    df = data_lv.loc[data_lv.Segm == num_seg]
    result = sm.ols(formula="Voltage ~ WT", data=df).fit()
    fig, ax = plt.subplots(nrows=2, figsize=(5, 7))
    fig.tight_layout(pad=3)
    #sns.pointplot(x='WT', y='Voltage', data=df, ax=ax[2])
    #ax[2].set_title('Conf. intervals')
    ax[0].set_title('Residuals')
    ax[1].set_title('Regression plot for {} segment'.format(num_seg))

    sns.distplot(result.resid, ax=ax[0])
    sns.regplot(x='WT', y='Voltage', data=df, ax=ax[1], x_jitter=.1, x_estimator=np.mean)

    box_text = 'r2 = {}\nslope = {}'.format(round(GET_R2(num_seg), 5), round(GET_SLOPE(num_seg), 5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.05, 0.05, box_text, transform=ax[1].transAxes, bbox=props)
    plt.savefig(os.path.join(out_dir, '{}.png'.format(num_seg)))
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
    parser.add_argument('-d', type=str, help='directory with *data*.csv')

    args = parser.parse_args()

    if args.f is None and args.d is None:
        raise ValueError('No path to data')
    elif args.d is None:
        data_lv = pd.read_csv(args.f, index_col=0)
        folder = args.f.split('/')[-1].split('.')[0]
        PLOT_DIR = os.path.join(PLOT_DIR, folder)
        if not os.path.isdir(PLOT_DIR):
            os.mkdir(PLOT_DIR)

        for n in tqdm(range(1, 18)):
            CREATE_STATS_PLOTS(n, PLOT_DIR)

    elif args.f is None:
        files = list(os.walk(args.d))[0][2]
        for f in tqdm(files):
            data_lv = pd.read_csv(os.path.join(args.d, f), index_col=0)
            folder = f.split('/')[-1].split('.')[0]
            out_dir = os.path.join(PLOT_DIR, folder)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            for n in range(1, 18):
                CREATE_STATS_PLOTS(n, out_dir)