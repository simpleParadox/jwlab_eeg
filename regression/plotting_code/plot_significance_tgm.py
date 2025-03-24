from curses.ascii import alt
from re import I
from turtle import color
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from mne.stats import fdr_correction, permutation_cluster_test
import glob
from tqdm import tqdm
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, Span, Label
from bokeh.io import export_png
from bokeh.io.export import get_svg
from selenium import webdriver
import chromedriver_binary
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import get_significance
from path_mappings import w2v_path_mapping_multiple_seeds, perm_w2v_path_mapping, ph_path_mapping_multiple_seeds, perm_ph_path_mapping
import pandas as pd


def plot_tgm_image(data, times, mask=None, ax=None, vmax=None, vmin=None,
               draw_mask=None, draw_contour=None, colorbar=True,
               draw_diag=True, draw_zerolines=True, xlabel="Time (s)", ylabel="Time (s)",
               cbar_unit="%", cmap="RdBu_r", mask_alpha=.5, mask_cmap="RdBu_r"):
    """Return fig and ax for further styling of GAT matrix, e.g., titles

    Parameters
    ----------
    data: array of scores
    times: list of epoched time points
    mask: None | array
    ...
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if vmax is None:
        vmax = np.abs(data).max()
    if vmin is None:
        vmax = np.abs(data).max()
        vmin = -vmax
    tmin, tmax = xlim = times[0], times[-1]
    extent = [tmin, tmax, tmin, tmax]
    im_args = dict(interpolation='nearest', origin='lower',
                   extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    if mask is not None:
        draw_mask = True if draw_mask is None else draw_mask
        draw_contour = True if draw_contour is None else draw_contour
    if any((draw_mask, draw_contour,)):
        if mask is None:
            raise ValueError("No mask to show!")

    if draw_mask:
        ax.imshow(data, alpha=mask_alpha, cmap=mask_cmap, **im_args)
        im = ax.imshow(np.ma.masked_where(~mask, data), cmap=cmap, **im_args)
    else:
        im = ax.imshow(data, cmap=cmap, **im_args)
    if draw_contour and np.unique(mask).size == 2:
        big_mask = np.kron(mask, np.ones((10, 10)))
        ax.contour(big_mask, colors=["k"], extent=extent, linewidths=[1], aspect=1, corner_mask=True, antialiased=False, levels=[.5], alpha=0.9)
    # plt.show()
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)

    if draw_diag:
        ax.plot((tmin, tmax), (tmin, tmax), color="k", linestyle=":")
    if draw_zerolines:
        ax.axhline(0, color="k", linestyle=":")
        ax.axvline(0, color="k", linestyle=":")

    if ylabel != '': ax.set_ylabel(ylabel)
    if xlabel != '': ax.set_xlabel(xlabel)
    #     plt.xticks(np.arange)
    #     ax.xaxis.set_tick_params(direction='out', which='bottom')
    #     ax.tick_params(axis='x',direction='out')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_unit, fontsize=12)
    ax.set_aspect(1. / ax.get_data_ratio())
    #     ax.set_title("GAT Matrix")

    return (fig if ax is None else ax), im


def perm_stat_fun1(X, all_acc_waves=[], p_val_thresh=0.05, sigma=0, method='negcorr', trainTask=3, testTask=3, wnum=0):
    # NOTE: I believe this is the one that should be used.
    # The kde is calculated on the permuted accuracies for ALL THE WINDOWS.
    pvalues = np.ones(X.shape)
    # tvalues = np.zeros(X.shape)
    # pvalues_fdr = np.ones(X.shape)  # after correction
    # reject_fdr = np.zeros(X.shape, dtype=bool)  # significant pvalues after fdr
    if all_acc_waves == []:
        return
    append_truth = False
    ptests = all_acc_waves
    ptests = np.ndarray.flatten(np.array(ptests))
    kde = stats.gaussian_kde(ptests)  # , bw_method='scott')
    # X = np.squeeze(X)
    for trainTime in range(pvalues.shape[0]):
        print(trainTime)
        for testTime in range(pvalues.shape[1]):
            # if append_truth:
            #     ptests = np.append(ptests, X[trainTime, testTime])
            #     kde = stats.gaussian_kde(ptests)  # , bw_method='scott')
            # return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])
            #             t_ob = (X[trainTime,testTime] - np.mean(ptests))/np.sqrt(var/ptests.shape[0])
            pv = kde.integrate_box_1d(X[trainTime, testTime], 1)
            #             print(pv)
            if pv == 0:
                pv = 1 / ptests.shape[0]
                # xs = np.linspace(0, 100, num=100)
            #                 plt.plot(xs, kde(xs))
            #                 plt.show()
            # if X[trainTime, testTime] > 59:
            #     pass
            #                 print(pv, X[trainTime, testTime], trainTime, testTime)
            pvalues[trainTime, testTime] = pv
    # fdr correction
    reject_fdr, pvalues_fdr = fdr_correction(np.squeeze(pvalues), alpha=p_val_thresh, method=method)
    if np.sum(reject_fdr) == 0:  # no significant acc
        threshold_fdr = 100
    else:
        threshold_fdr = np.min(np.abs(X)[reject_fdr])

    return reject_fdr, threshold_fdr


permuted_tgm_accs_path = None
observed_tgm_accs_path = None


all_acc_waves = []
for file in glob.glob(permuted_tgm_accs_path + '/*.npz'):
    all_acc_waves.append(np.load(file))
    all_acc_waves.append(np.load(file, allow_pickle=True)['arr_0'][:-9, :-9])
all_acc_waves = np.array(all_acc_waves)


truth_tgm_csv = pd.read_csv(observed_tgm_accs_path)
truth = truth_tgm_csv.iloc[:-9, 1:-9].values # This is the numpy array and should go into the perm_stat_fun1 function for 'X'.
reject_fdr, threshold_fdr = perm_stat_fun1(truth, all_acc_waves=all_acc_waves, p_val_thresh=0.05, sigma=0, method='relative', trainTask=3, testTask=3, wnum=0)
print(threshold_fdr)

times = [-100, 1000]

# Plot tgm
plax, im = plot_tgm_image(truth, times, mask=truth > threshold_fdr, vmax=np.max(truth), vmin=np.min(truth), 
               draw_mask=True, draw_contour=True, colorbar=True, draw_diag=True, draw_zerolines=True, 
               xlabel="Time (s)", ylabel="Time (s)", cbar_unit="%", cmap="RdBu_r", mask_alpha=.5, mask_cmap="RdBu_r")
plax.set_title("TGM")
plt.savefig('tgm_test.png')
