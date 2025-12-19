import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import mkdir
from utils.processing import open_fits_catalog
import os

def main():


    MATCHED_DIR = 'data/szcat/'
    DATASET_DIR = mkdir('data/dlcat/')
    PLOT_DIR = mkdir('figures/')

    data = open_fits_catalog(os.path.join(MATCHED_DIR, "szcat.fits"))


    z_mask = z_cuts_mask(cat=data, zlow=0.01, zhigh=1, zcol_name='Z')
    mag_mask = mag_cuts_mask(cat=data)
    color_mask = color_cuts_mask(cat=data)
    snr_mask = snr_cuts_mask(cat=data)

    all_mask = color_mask*z_mask*mag_mask*snr_mask 
    with open(os.path.join(DATASET_DIR, "dlcat_summary.txt"), 'w') as f:
        f.write(f'Pre-Cut data = {len(z_mask):} Objects')
        f.write(f'percentual of good z = {np.sum(z_mask)/len(z_mask)*100:.2f} %')
        f.write(f'percentual of good mag (g,r,i,z) = {np.sum(mag_mask)/len(mag_mask)*100:.2f} %')
        f.write(f'percentual of good color (g-r), (r-i), (i-z) = {np.sum(color_mask)/len(color_mask)*100:.2f} %')
        f.write(f'percentual of good SNR = {np.sum(snr_mask)/len(snr_mask)*100:.2f} %')
        f.write(f'percentual of surviving  objects = {np.sum(all_mask)/len(all_mask)*100:.2f} %-> {np.sum(all_mask)} Objects')


    data = data# for non-test version: data[all_mask]

    #########SHUFFLING##################
    np.random.seed(137)
    data = data.copy()[np.random.choice(len(data), len(data), replace=False)]

    features = ['RA', 'DEC', 'QUICK_OBJECT_ID',
                'MAG_AUTO_G', 'MAGERR_AUTO_G',
                'MAG_AUTO_R', 'MAGERR_AUTO_R',
                'MAG_AUTO_I', 'MAGERR_AUTO_I',
                'MAG_AUTO_Z', 'MAGERR_AUTO_Z', 'Z']

    data = data[features].to_pandas()
    data.to_csv(os.path.join(DATASET_DIR, 'dlcat.csv'))

    # if already built: 
    # data = pd.read_csv('data/dlcat.csv')

    zwidth=0.05
    flat_z_lim = 0.8 # for DLTRAIN-A; 0.6 for DLTRAIN-B; and 0.0 for DLTRAIN-C

    bins = np.arange(0.01, 1, zwidth)

    np.random.seed(137)
    test_mask = np.random.uniform(0,1,len(data)) < .2

    test_data = data[test_mask]
    trainable_data = data[~test_mask]

    plt_style()
    plt.figure(figsize=(8,8), dpi=100)

    # print(np.arange(0, 1, zwidth))
    # aaa
    flat_mask = flat_hist_masks(zdata=trainable_data['Z'], flat_z_lim=flat_z_lim, 
                                zwidth=zwidth, zmin=0, zmax=1)#np.full(len(trainable_data), True)#

    train_data = trainable_data[flat_mask]
    extra_data = trainable_data[~flat_mask]
    plt.hist(train_data['Z'], bins=bins, alpha=.5, label='train')
    plt.hist(test_data['Z'], bins=bins, alpha=.5, label='test')
    plt.hist(extra_data['Z'], bins=bins, alpha=.5, label='extra')
    plt.xlabel('$z_{spec}$')
    plt.ylabel('count')
    plt.savefig(os.path.join(PLOT_DIR, "dltrain_a_hist.png"), bbox_inches="tight")


    bandas = ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z']#, 'MAG_W1', 'MAG_W2', 'MAG_W3', 'MAG_W4']

    for i in range(len(bandas)):
        for b in bandas[i+1:]:
            train_data[f'{bandas[i]}-{b}'] = np.array(train_data[f'{bandas[i]}']) - np.array(train_data[f'{b}']) 
            test_data[f'{bandas[i]}-{b}'] = np.array(test_data[f'{bandas[i]}']) - np.array(test_data[f'{b}'])
            extra_data[f'{bandas[i]}-{b}'] = np.array(extra_data[f'{bandas[i]}']) - np.array(extra_data[f'{b}'])

    train_data.to_csv(DATASET_DIR + 'dltrain_a.csv')
    test_data.to_csv(DATASET_DIR + 'test_data.csv')
    extra_data.to_csv(DATASET_DIR + 'extra_data.csv')

    return


####### Functions #######
def z_cuts_mask(cat, zlow=0.01, zhigh=1.5, zcol_name='Z' ):
    mask = np.array(cat[zcol_name]>zlow) & np.array(cat[zcol_name]<zhigh)
    return mask

def mag_cuts_mask(cat, magg_key='MAG_G', magr_key='MAG_R',
                     magi_key='MAG_I', magz_key='MAG_Z',
                     magw1_key='MAG_W1', magw2_key='MAG_W2',
                     magw3_key='MAG_W3', magw4_key='MAG_W4'):
    # https://arxiv.org/abs/2211.09492v1
    g_mask = cat[magg_key.replace('MAG', 'MAG_AUTO')]<22.5
    r_mask = cat[magr_key.replace('MAG', 'MAG_AUTO')]<22.5
    i_mask = cat[magi_key.replace('MAG', 'MAG_AUTO')]<22.5
    z_mask = cat[magz_key.replace('MAG', 'MAG_AUTO')]<22.5
    
    all_mask = g_mask*r_mask*z_mask*i_mask
    return all_mask
    
def snr_cuts_mask(cat, magg_key='MAG_G', magr_key='MAG_R',
                     magi_key='MAG_I', magz_key='MAG_Z',
                     magw1_key='MAG_W1', magw2_key='MAG_W2',
                     magw3_key='MAG_W3', magw4_key='MAG_W4'):
    
    magi_err = magi_key.replace('MAG', 'MAGERR_AUTO')
    magr_err = magr_key.replace('MAG', 'MAGERR_AUTO')
    magg_err = magg_key.replace('MAG', 'MAGERR_AUTO')
    magz_err = magz_key.replace('MAG', 'MAGERR_AUTO')
    
    snr_g = 1.0875/np.array(cat.field(magg_err)) > 3
    snr_r = 1.0875/np.array(cat.field(magr_err)) > 5
    snr_i = 1.0875/np.array(cat.field(magi_err)) > 5
    snr_z = 1.0875/np.array(cat.field(magz_err)) > 5
    
    mask = snr_g & snr_r & snr_i & snr_z
    # cat_out = cat_table[snr_g & snr_r & snr_i & snr_z]
    return mask

def color_cuts_mask(cat, magg_key='MAG_AUTO_G', magr_key='MAG_AUTO_R',
                     magi_key='MAG_AUTO_I', magz_key='MAG_AUTO_Z'):
    
    # https://arxiv.org/abs/1708.01531
    
    gr_mask = (cat[magg_key] - cat[magr_key] > -1) & (cat[magg_key] - cat[magr_key] < 4)
    ri_mask = (cat[magr_key] - cat[magi_key] > -1) & (cat[magr_key] - cat[magi_key] < 4)
    iz_mask = (cat[magi_key] - cat[magz_key] > -1) & (cat[magi_key] - cat[magz_key] < 4)
    
    all_mask = gr_mask*ri_mask*iz_mask
    return all_mask

def flat_hist_masks(zdata, flat_z_lim=0.8, zwidth=0.05, zmin=None, zmax=None):
    if zmin==None:
        aaa
        zmin = np.min(zdata)
    if zmax==None:
        zmax = np.max(zdata)
    
    # print(zmin)
    zbins = np.arange(zmin, zmax, zwidth)
    # aaaaaaa
    # print(zdata)
    zidxs = np.array(range(len(zdata)))
    
    height_lim = min([np.sum((zdata>aux_z_lim-zwidth) & (zdata<aux_z_lim)) for aux_z_lim in np.arange(zmin+zwidth, flat_z_lim+zwidth, zwidth)])
    
    flat_idxs = []
    
    for zlim in zbins:
        
        zbin_mask = (zdata>zlim) & (zdata<=zlim+zwidth)
        if np.sum(zbin_mask)>=height_lim:
            flat_idxs += list(np.random.choice(zidxs[zbin_mask], height_lim, replace=False))
        else:
            flat_idxs += list(zidxs[zbin_mask])
    flat_mask = np.in1d(zidxs, flat_idxs)
    # extra_mask = ~flat_mask
    
    return flat_mask#, extra_mask
    
def plt_style():
    plt.rcParams.update({
                        'lines.linewidth':1.0,
                        'lines.linestyle':'-',
                        'lines.color':'black',
                        'font.family':'serif',
                        'font.weight':'normal',
                        'font.size':22.0,
                        'text.color':'black',
                        'text.usetex':False,
                        'axes.edgecolor':'black',
                        'axes.linewidth':1.0,
                        'axes.grid':False,
                        'axes.titlesize':'x-large',
                        'axes.labelsize':'x-large',
                        'axes.labelweight':'normal',
                        'axes.labelcolor':'black',
                        'axes.formatter.limits':[-4,4],
                        'xtick.major.size':7,
                        'xtick.minor.size':4,
                        'xtick.major.pad':8,
                        'xtick.minor.pad':8,
                        'xtick.labelsize':'medium',
                        'xtick.minor.width':1.0,
                        'xtick.major.width':1.0,
                        'ytick.major.size':7,
                        'ytick.minor.size':4,
                        'ytick.major.pad':8,
                        'ytick.minor.pad':8,
                        'ytick.labelsize':'medium',
                        'ytick.minor.width':1.0,
                        'ytick.major.width':1.0,
                        'legend.numpoints':1,
                        #'legend.fontsize':'x-large',
                        'legend.shadow':False,
                        'legend.frameon':False})


if __name__=="__main__":
    main()