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
    
def clean(specz_legacy_file_path):
    
    desi_filename = specz_legacy_file_path.replace('.fits', 'spec_match_DESI.fits')
    desi_match = open_fits_catalog(desi_filename)

    julia_filename = specz_legacy_file_path.replace('.fits', 'spec_match_JULIA.fits')
    julia_match = open_fits_catalog(julia_filename)

    erik_filename = specz_legacy_file_path.replace('.fits', 'spec_match_ERIK.fits')
    erik_match = open_fits_catalog(erik_filename)
    # brick_lits = 
    
    
    
    
    unique_desi = np.unique(desi_match['QUICK_OBJECT_ID'], return_index=True)[1]
    desi_match = desi_match[unique_desi]
    
    # taking the ones in JULIA that are not in DESI
    unique_julia_ids = np.unique(julia_match['QUICK_OBJECT_ID'])
    unique_julia_idxs = np.unique(julia_match['QUICK_OBJECT_ID'], return_index=True)[1]

    notin_desi_julia_unique_idxs = unique_julia_idxs[~np.in1d(unique_julia_ids, desi_match['QUICK_OBJECT_ID'])]
    
    julia_match = julia_match[notin_desi_julia_unique_idxs]
    # end
    
    # taking the ones in ERIK that are not in (DESI & JULIA)
    unique_erik_ids = np.unique(erik_match['QUICK_OBJECT_ID'])
    unique_erik_idxs = np.unique(erik_match['QUICK_OBJECT_ID'], return_index=True)[1]

    notin_desijulia_erik_unique_idxs = unique_erik_idxs[~(np.in1d(unique_erik_ids, desi_match['QUICK_OBJECT_ID'])|np.in1d(unique_erik_ids, julia_match['QUICK_OBJECT_ID']))]

    erik_match = erik_match[notin_desijulia_erik_unique_idxs]
    #end
    desi_match['match_source'] = Table.Column(['DESI']*len(desi_match), dtype=str)
    julia_match['match_source'] = Table.Column(['JULIA']*len(julia_match), dtype=str)
    erik_match['match_source'] = Table.Column(['ERIK']*len(erik_match), dtype=str)
    
    erik_match['Z'] = erik_match['z']
    desi_match['Z'] = desi_match['z']
    desi_match['RA'] = desi_match['mean_fiber_ra']
    desi_match['DEC'] = desi_match['mean_fiber_dec']
    erik_match['RA'] = erik_match['RA_1']
    erik_match['DEC'] = erik_match['DEC_1']
    julia_match['RA'] = julia_match['RA_1']
    julia_match['DEC'] = julia_match['DEC_1']
    
    julia_match['ZERR'] = julia_match['ERR_Z']
    erik_match['ZERR'] = erik_match['e_z']
    desi_match['ZERR'] = np.zeros(len(desi_match))#erik_match['e_z']
    match_columns = list(np.array(list(julia_match.columns))[np.in1d(list(julia_match.columns), list(desi_match.columns))])

    desi_match = desi_match[match_columns]
    julia_match = julia_match[match_columns]
    erik_match = erik_match[match_columns]

    final_match = vstack([julia_match, desi_match, erik_match])#vstack(brick_match_lists)
    final_match.write(specz_legacy_file_path.replace('.fits', 'spec_match_MERGED.fits'), format='fits', overwrite=True)
        
        
        
MATCHED_DIR = mkdir('/tf/astrodados/DELVE_DR2_Xmatch_specz/')   
DELVE_DR2_PATH = '/tf/astrodados/DELVE_DR2_cleaned/'
DELVE_DR2_FILES = [MATCHED_DIR + file for file in os.listdir(DELVE_DR2_PATH) if file.endswith('.fits')]

NUM_PROCESS = 6
Parallel(n_jobs=NUM_PROCESS)(delayed(clean)(specz_legacy_file_path=DELVE_DR2_FILES[i]) for i in tqdm(range(len(DELVE_DR2_FILES))))
#DOING MERGE
