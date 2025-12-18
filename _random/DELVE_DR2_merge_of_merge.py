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
    Return de table version of the fits catalog.

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


MATCHED_DIR = mkdir('/tf/astrodados/DELVE_DR2_Xmatch_specz/')   
# LEGACY_DR10_PATH = '/tf/astrodados/portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.0/'
LEGACY_DR10_SPEC_FILES = [MATCHED_DIR + file for file in os.listdir(MATCHED_DIR) if file.endswith('MERGED.fits')]

merge_list = []

for legacy_spec_file in tqdm(LEGACY_DR10_SPEC_FILES):
    merge_list.append(open_fits_catalog(legacy_spec_file))

merge_of_merges = vstack(merge_list)
merge_of_merges.write(MATCHED_DIR+'DELVE_SPECZ_MATCH_18SET2023.fits', format='fits', overwrite=True)