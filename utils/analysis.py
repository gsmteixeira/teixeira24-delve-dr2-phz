import sys
import pickle 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from utils.beauty import plt_style
from tqdm import tqdm
import os
import sys


def h2dplot(x, y, ax=None, nbins=50, size=.5, alpha=.5, marker='.'):


    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = nbins)

    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data ,
                np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    cax=ax.scatter( x, y, c=z,s=size,alpha=alpha, marker=marker)

    return ax, cax 

def metrics_per_bin(zspec, zphot, zwidth,rmag=None, magwidth=None,xaxis='zspec', zlim=1):
    
    
    Bins_Z   = np.arange(0, zlim+zwidth, zwidth)
    if xaxis=='rmag':
        Bins_Z   = np.arange(np.min(rmag), np.max(rmag)+magwidth, magwidth)
    
    metrics_list = ['$\sigma_{NMAD}$','mean bias', 'median bias', 'outfrac']#, 's68']
    
    metrics = {}
    for m in metrics_list:
        metrics[m] = []
    
    for bin_val in Bins_Z:
        
        if xaxis=='zspec':
            between = (zspec>=bin_val) & (zspec<bin_val+zwidth)
            
        elif xaxis=='zphot':
            between = (zphot>=bin_val) & (zphot<bin_val+zwidth)
            
        elif xaxis=='rmag':
            between = (rmag>=bin_val) & (rmag<bin_val+magwidth)
        
        zspec_bin = zspec[between]
        zphot_bin = zphot[between]

        delta_z = zphot_bin - zspec_bin#zspec_bin - zphot_bin
        
        snmad = 1.48 * np.median(np.absolute(delta_z - np.median(delta_z)) / (1+zspec_bin))
        #https://iopscience.iop.org/article/10.1086/591786/pdf
        
        mean_bias = np.mean(delta_z, axis=0)

        median_bias = np.median(delta_z)

        outlier_frac_point = np.sum(  np.abs(delta_z)/(1+zspec_bin) >0.15)/len(delta_z)
        
        # p16 = np.percentile(delta_z, 15.9)
        # p84 = np.percentile(delta_z, 84.1)
        # s68 = (p84-p16)/2
        
        
        metrics['$\sigma_{NMAD}$'].append(snmad)
        metrics['mean bias'].append(mean_bias)
        metrics['median bias'].append(median_bias)
        metrics['outfrac'].append(outlier_frac_point)
        # metrics['s68'].append(s68)
    
    return metrics, Bins_Z
