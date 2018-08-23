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

PLOT_DIR = os.path.join(HOME, 'source/presentationEAVM/segments_plot')

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

    for n in tqdm(range(1, 18)):
        CREATE_STATS_PLOTS(n)

    s_apex = ColumnDataSource(data=dict(name=['17'], img=['17.png'], r2=[GET_R2(17)], slope=[GET_SLOPE(17)]))

    s_apical = [ColumnDataSource(data=dict(
        name=['{}'.format(12 + i)],
        img=['{}.png'.format(12 + i)],
        r2=[GET_R2(12 + i)],
        slope=[GET_SLOPE(12 + i)])) for i in [4, 1, 2, 3]]

    s_basal_1 = [ColumnDataSource(data=dict(
        name=['{}'.format(6 + i)], img=['{}.png'.format(6 + i)],
        r2=[GET_R2(6 + i)],
        slope=[GET_SLOPE(6 + i)])) for i in [6, 1, 2, 3, 4, 5]]

    s_basal_2 = [ColumnDataSource(data=dict(
        name=['{}'.format(i)],
        img=['{}.png'.format(i)],
        r2=[GET_R2(i)],
        slope=[GET_SLOPE(i)])) for i in [6, 1, 2, 3, 4, 5]]

    p = figure(plot_width=500, plot_height=700, x_range=(-10, 10), y_range=(-10, 10))

    r0 = 2
    rd = 2
    p.axis.visible = False
    p.ellipse([0], [0], width=20, height=11, height_units='data', width_units='data', fill_color='white',
              line_width=6)
    p.patch([0, 10, 10, 0, 0], [10, 10, -10, -10, 10], fill_color='white', line_color='white')
    p.wedge(x=0, y=0, radius=r0, start_angle=RAD(0), end_angle=RAD(360), color="firebrick", alpha=0.6, source=s_apex)
    for i in range(0, 4):
        p.annular_wedge(x=0, y=0, inner_radius=r0, outer_radius=r0 + rd, start_angle=RAD(-45 + i * 90),
                        end_angle=RAD(45 + i * 90), color="green", alpha=0.6,
                        source=s_apical[i])
    r0 += rd
    for i in range(6):
        p.annular_wedge(x=0, y=0, inner_radius=r0, outer_radius=r0 + rd, start_angle=RAD(0 + i * 60),
                        end_angle=RAD(60 + i * 60), color="red", alpha=0.6,
                        source=s_basal_1[i])
    r0 += rd
    for i in range(6):
        p.annular_wedge(x=0, y=0, inner_radius=r0, outer_radius=r0 + rd, start_angle=RAD(0 + i * 60),
                        end_angle=RAD(60 + i * 60), color="blue", alpha=0.6,
                        source=s_basal_2[i])

    p.add_tools(HoverTool(tooltips="""
        <div>
            <div>
                <span style="font-size: 12;"><b>Segment</b>: @name</span><br />
            </div>
            <div>
                <img
                    src="@img" height="360" alt="@img" width="120"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 12;"><b>R2</b>: @r2</span><br />
                <span style="font-size: 12;"><b>slope</b>: @slope</span><br />
            </div>
        </div>
        """
                          ))
    output_file(os.path.join(PLOT_DIR, "plot.html"))
    show(p, notebook_handle=False)
