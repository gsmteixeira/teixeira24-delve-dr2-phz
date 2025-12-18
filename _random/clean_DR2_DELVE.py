import sys
import astropy_healpix as ah
from mocpy import MOC
from mocpy import World2ScreenMPL
import astropy_healpix as ah
import healpy as hp
from astropy.io import fits
import numpy as np
import os
import time
from astropy.coordinates import SkyCoord,Angle
import astropy.coordinates as coord
from astropy import units as u
from astropy.table import Table
from astropy.table import hstack, vstack
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def mkdir(directory_path): 
    if os.path.exists(directory_path): 
        return directory_path
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return sys.exit("Erro ao criar diretório") 
        return directory_path

def open_fits_catalog(fits_file):
    """
    Return the table version of the fits catalog.

    Parameters:
    fits_file (str): catalog path.
    
    Returns:
    fits Table: Table catalog 
    """
    
    hdu_list=fits.open(fits_file, ignore_missing_end=True)
    #print hdu_list
    hdu = hdu_list[1]    # table extensions can't be the first extension, so there's a dummy image extension at 0
    #print hdu.header
    cat_table = Table(hdu.data)
    cols=hdu.columns
    return cat_table

def modest_class_mask(catalog_,classes_list=[1,3],sm_key='SPREAD_MODEL_G',smerr_key='SPREADERR_MODEL_G',
                 mag_key='MAG_AUTO_G',wsm_key='WAVG_SPREAD_MODEL_G',
                 return_class='mask'):
    
    """
    MODEST_CLASS star-galaxy classification. Return a mask for the desired classes or an array with the respective classes.
    
    Parameters:
    catalog_ (fits Table): Table catalog
    classes_list (list): desired classes to be masked as True
    sm_key (str): SPREAD_MODEL_{G,R,I,Z} colum name
    smerr_key (str): SPREADERR_MODEL_{G,R,I,Z} colum name
    mag_key (str): MAG_AUTO_{G,R,I,Z} colum name
    wsm_key (str): WAVG_SPREAD_MODEL_{G,R,I,Z} colum name
    return_class (str): 'class' for all the respective classes, 'mask' for the desired classes mask, 'both' for the classes an the mask for the ones of interest
    
    Returns:
    numpy.array or tuple: classes array, mask array or (classes array, mask array) depending on the return_class options, respectively 
    
    The definitions can be seen in table 6 of https://arxiv.org/pdf/1708.01531.pdf
    """
    
    SPREAD = np.array(catalog_[sm_key])
    SPREADERR = np.array(catalog_[smerr_key])
    MAG = np.array(catalog_[mag_key])
    WSM = np.array(catalog_[wsm_key])
    # print(type(SPREAD), type(SPREADERR), type(MAG), type(WSM))
        
    mtype_dic = {'likstar':0 ,
                 'highgal':1 ,
                 'highstar':2 ,
                 'ambigous':3 }
    classes = np.full(len(catalog_), 99).astype('int16')

    for mt in mtype_dic:
        if mt == 'likstar':
            selection = SPREAD + (5/3)*SPREADERR < -0.002
            classes[selection] = mtype_dic[mt]

        elif mt == 'highgal':
            selection = (SPREAD + (5/3)*SPREADERR >0.005) & ~( (np.abs(WSM) < 0.002) & (MAG < 21.5) )
            classes[selection] = mtype_dic[mt]

        elif mt == 'highstar':
            selection = SPREAD + (5/3)*SPREADERR < 0.002
            classes[selection] = mtype_dic[mt]

        elif mt == 'ambigous':
            selection = (SPREAD + (5/3)*SPREADERR > 0.002) & (SPREAD + (5/3)*SPREADERR < 0.005)
            classes[selection] = mtype_dic[mt]
        
    if return_class=='mask':     
        mask = np.full(len(catalog_), False)

        for clas in  classes_list:
            mask += classes == clas

        return mask
    
    elif return_class=='class':

        return classes

    elif return_class=='both':     
        mask = np.full(len(catalog_), False)

        for clas in  classes_list:
            mask += classes == clas

        return classes, mask

def mag_g_mask(cat_, maglim=23.5,g_key='MAG_AUTO_G'): 
    
    mask = np.array(cat_[g_key])<maglim
    
    return mask

def flag_g_mask(cat_, flag_key='FLAGS_G', flaglim=3): 
    
    mask = np.array(cat_[flag_key])<flaglim
    
    return mask

def clean_catalog(file_path, save_dir):
    
    cat = open_fits_catalog(file_path)
    
    do_mag_g_mask =  mag_g_mask(cat, maglim=23.5,g_key='MAG_AUTO_G')
    do_flag_g_mask =  flag_g_mask(cat, flag_key='FLAGS_G', flaglim=3)
    do_modest_class_mask = modest_class_mask(cat,classes_list=[1,3],
                                             sm_key='SPREAD_MODEL_G',
                                             smerr_key='SPREADERR_MODEL_G',
                                             mag_key='MAG_AUTO_G',
                                             wsm_key='WAVG_SPREAD_MODEL_G',
                                             return_class='mask')
    
    final_mask = do_mag_g_mask*do_flag_g_mask*do_modest_class_mask
    
    filename = file_path.split('/')[-1]
    
    with open(save_dir+'delve_cleaning_summary.txt', 'a') as f:
        f.write(f'{filename} - initial:{len(cat):010d} objs - final:{sum(final_mask):010d} objs\n')
        f.close()
    
    final_cat = cat[final_mask]
    final_cat.write(save_dir+filename, format='fits', overwrite=True)
    

DELVE_DR2_R1_PATH = '/tf/dados10T/delve_dr2_cats/r1/'
DELVE_DR2_R2_PATH = '/tf/dados10T/delve_dr2_cats/r2/'
DELVE_DR2_FILES = [DELVE_DR2_R1_PATH + file for file in os.listdir(DELVE_DR2_R1_PATH) if file.endswith('.fits')] + \
                  [DELVE_DR2_R2_PATH + file for file in os.listdir(DELVE_DR2_R2_PATH) if file.endswith('.fits')]

SAVE_DIR =  mkdir('/tf/astrodados/DELVE_DR2_cleaned/')
NUM_PROCESS = 6

with open(SAVE_DIR + 'delve_cleaning_summary.txt', 'w') as f:
    # f.write(f'')
    f.close()

start_clean = time.time()
Parallel(n_jobs=NUM_PROCESS)(delayed(clean_catalog)(file_path=DELVE_DR2_FILES[i], save_dir=SAVE_DIR) for i in tqdm(range(len(DELVE_DR2_FILES))))
end_clean = time.time()

with open(SAVE_DIR + 'delve_cleaning_summary.txt', 'a') as f:
    f.write(f'ELAPSED TIME = {(end_clean - start_clean)/60:.2f} min\n')
    f.close()