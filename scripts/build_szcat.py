"""
Author: Gabriel Teixeira

Pipeline for photometric–spectroscopic cross-matching of DELVE DR2 catalogs.
Colum names may be deprecated.

This script performs parallel cross-matching between DELVE photometric catalogs
and multiple spectroscopic reference catalogs, resolves
duplicate matches, and produces a final merged spectroscopic redshift catalog.
"""

import os
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.utils import mkdir
from utils.processing import open_fits_catalog, match_cats
from astropy.table import Table, vstack
import pandas as pd
import shutil


def main():
    """
    Main execution function.

    This function:
    - Loads DELVE DR2 photometric catalogs
    - Performs parallel cross-matching with spectroscopic reference catalogs
    - Merges and cleans matched catalogs
    - Produces a final spectroscopic redshift catalog
    - Removes temporary working directories upon completion
    """
    DELVE_CLEANED_PATH =  'data/mcat/'
    DELVE_DR2_FILES = [os.path.join(DELVE_CLEANED_PATH, file) for file in os.listdir(DELVE_CLEANED_PATH) if file.endswith('.fits')]

    MATCHED_DIR_TEMP = mkdir('data/match.temp/')
    MATCHED_DIR = mkdir('data/szcat/')

    DESI_DR1_PATH = '/tf/astrodados/SPEC_Z/DESi_EDR' #https://data.desi.lbl.gov/public/edr/
    JULIA_CATALOG_PATH = '/tf/astrodados/SPEC_Z/julia_compilation' #https://doi.org/10.1016/j.ascom.2018.08.008
    ERIK_CATALOG_PATH = '/tf/astrodados/SPEC_Z/erik_compilation' #https://doi. org/10.5281/zenodo.11641315

    # concatenated version of the catalogs
    DESI_CATALOG_FILES = [os.path.join(DESI_DR1_PATH, 'DESi_EDR.fits')]
    JULIA_CATALOG_FILES = [os.path.join(JULIA_CATALOG_PATH, 'SPECZ_24NOV20.fits')]
    ERIK_CATALOG_FILES = [os.path.join(ERIK_CATALOG_PATH, 'SpecZ_Catalogue_20230704.csv')]

    NUM_PROCESS = 6

    do_matches(NUM_PROCESS=NUM_PROCESS,
               MATCHED_DIR=MATCHED_DIR_TEMP,
               DELVE_DR2_FILES=DELVE_DR2_FILES,
               DESI_CATALOG_FILES=DESI_CATALOG_FILES,
               JULIA_CATALOG_FILES=JULIA_CATALOG_FILES,
               ERIK_CATALOG_FILES=ERIK_CATALOG_FILES)
    
    do_merge(NUM_PROCESS=NUM_PROCESS,
             MATCHED_DIR= MATCHED_DIR_TEMP,
             DELVE_DR2_PATH=DELVE_CLEANED_PATH)

    do_szcat(MATCHED_DIR_TEMP=MATCHED_DIR_TEMP,
             MATCHED_DIR=MATCHED_DIR)

    if os.path.exists(MATCHED_DIR_TEMP):
        shutil.rmtree(MATCHED_DIR_TEMP)
    
    return


###############  Functions  ############### 


def do_szcat(MATCHED_DIR_TEMP,
             MATCHED_DIR):
    """
    Build the final spectroscopic redshift catalog.

    Parameters
    ----------
    MATCHED_DIR_TEMP : str
        Path to the directory containing merged per-brick spectroscopic matches.
    MATCHED_DIR : str
        Output directory where the final stacked catalog is written.
    """
    DELVE_DR2_MERGED_SPEC_FILES = [MATCHED_DIR_TEMP + file for file in os.listdir(MATCHED_DIR_TEMP) if file.endswith('MERGED.fits')]
    merge_list = []
    for spec_file in tqdm(DELVE_DR2_MERGED_SPEC_FILES):
        merge_list.append(open_fits_catalog(spec_file))

    merge_of_merges = vstack(merge_list)
    merge_of_merges.write(MATCHED_DIR+'szcat.fits', format='fits', overwrite=True)


def do_merge(NUM_PROCESS,
            MATCHED_DIR,
            DELVE_DR2_PATH):
    """
    Merge spectroscopic matches for each DELVE photometric catalog.

    Parameters
    ----------
    NUM_PROCESS : int
        Number of parallel processes.
    MATCHED_DIR : str
        Directory containing cross-matched catalogs.
    DELVE_DR2_PATH : str
        Path to the original DELVE DR2 photometric catalogs.
    """
    DELVE_DR2_XMATCH_FILES = [MATCHED_DIR + file for file in os.listdir(DELVE_DR2_PATH) if file.endswith('.fits')]
    NUM_PROCESS = NUM_PROCESS
    Parallel(n_jobs=NUM_PROCESS)(delayed(clean)(specz_legacy_file_path=DELVE_DR2_XMATCH_FILES[i]) for i in tqdm(range(len(DELVE_DR2_XMATCH_FILES))))
 

def do_matches(NUM_PROCESS, 
               MATCHED_DIR,
               DELVE_DR2_FILES,
               DESI_CATALOG_FILES,
               JULIA_CATALOG_FILES,
               ERIK_CATALOG_FILES):
    """
    Perform parallel cross-matching between photometric and spectroscopic catalogs.

    Parameters
    ----------
    NUM_PROCESS : int
        Number of parallel processes.
    MATCHED_DIR : str
        Directory to store intermediate matched catalogs.
    DELVE_DR2_FILES : list of str
        List of DELVE DR2 photometric catalog paths.
    DESI_CATALOG_FILES : list of str
        List containing the DESI spectroscopic catalog path.
    JULIA_CATALOG_FILES : list of str
        List containing the JULIA spectroscopic catalog path.
    ERIK_CATALOG_FILES : list of str
        List containing the ERIK spectroscopic catalog path.
    """
    spec_num_list = []
    delve_num_list = []
    NUM_PROCESS = NUM_PROCESS

    delve_names = []
    desi_data = open_fits_catalog(DESI_CATALOG_FILES[0])#[:1000]
    julia_data = open_fits_catalog(JULIA_CATALOG_FILES[0])#[:1000]
    erik_data = Table.from_pandas(pd.read_csv(ERIK_CATALOG_FILES[0])[['RA','DEC','z', 'e_z', 'class_spec']])#[:1000]

    start_desi = time.time()
    Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(phot_path=DELVE_DR2_FILES[i], spec_cat=desi_data, save_dir=MATCHED_DIR, spec_name='DESI') for i in tqdm(range(len(DELVE_DR2_FILES))))
    end_desi = time.time()    
    with open(MATCHED_DIR+'DESI_match_time.txt', 'w') as f:
            f.write(f'elapsed time doing DESI match = {(end_desi-start_desi)/60:.2f} min\n')
            f.close()

    start_julia = time.time()
    Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(phot_path=DELVE_DR2_FILES[i], spec_cat=julia_data, save_dir=MATCHED_DIR, spec_name='JULIA') for i in tqdm(range(len(DELVE_DR2_FILES))))
    end_julia = time.time()    
    with open(MATCHED_DIR+'julia_match_time.txt', 'w') as f:
            f.write(f'elapsed time doing julia match = {(end_julia-start_julia)/60:.2f} min\n')
            f.close()

    start_erik = time.time()
    Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(phot_path=DELVE_DR2_FILES[i], spec_cat=erik_data, save_dir=MATCHED_DIR, spec_name='ERIK') for i in tqdm(range(len(DELVE_DR2_FILES))))
    end_erik = time.time()    
    with open(MATCHED_DIR+'ERIK_match_time.txt', 'w') as f:
            f.write(f'elapsed time doing ERIK match = {(end_erik-start_erik)/60:.2f} min\n')
            f.close()


def clean(specz_legacy_file_path):
    """
    Clean and merge spectroscopic matches from different reference catalogs.

    Parameters
    ----------
    specz_legacy_file_path : str
        Path to the base photometric catalog used for matching.
    """
    desi_filename = specz_legacy_file_path.replace('.fits', 'spec_match_DESI.fits')
    desi_match = open_fits_catalog(desi_filename)

    julia_filename = specz_legacy_file_path.replace('.fits', 'spec_match_JULIA.fits')
    julia_match = open_fits_catalog(julia_filename)

    erik_filename = specz_legacy_file_path.replace('.fits', 'spec_match_ERIK.fits')
    erik_match = open_fits_catalog(erik_filename)

    unique_desi = np.unique(desi_match['QUICK_OBJECT_ID'], return_index=True)[1]
    desi_match = desi_match[unique_desi]
    
    unique_julia_ids = np.unique(julia_match['QUICK_OBJECT_ID'])
    unique_julia_idxs = np.unique(julia_match['QUICK_OBJECT_ID'], return_index=True)[1]
    notin_desi_julia_unique_idxs = unique_julia_idxs[~np.in1d(unique_julia_ids, desi_match['QUICK_OBJECT_ID'])]
    julia_match = julia_match[notin_desi_julia_unique_idxs]
    
    unique_erik_ids = np.unique(erik_match['QUICK_OBJECT_ID'])
    unique_erik_idxs = np.unique(erik_match['QUICK_OBJECT_ID'], return_index=True)[1]
    notin_desijulia_erik_unique_idxs = unique_erik_idxs[~(np.in1d(unique_erik_ids, desi_match['QUICK_OBJECT_ID'])|np.in1d(unique_erik_ids, julia_match['QUICK_OBJECT_ID']))]
    erik_match = erik_match[notin_desijulia_erik_unique_idxs]

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
    desi_match['ZERR'] = np.zeros(len(desi_match))

    match_columns = list(np.array(list(julia_match.columns))[np.in1d(list(julia_match.columns), list(desi_match.columns))])

    desi_match = desi_match[match_columns]
    julia_match = julia_match[match_columns]
    erik_match = erik_match[match_columns]

    final_match = vstack([julia_match, desi_match, erik_match])
    final_match.write(specz_legacy_file_path.replace('.fits', 'spec_match_MERGED.fits'), format='fits', overwrite=True)


def match_loop(spec_cat, phot_path, save_dir, spec_name):
    """
    Match a single photometric catalog against a spectroscopic reference catalog.

    Parameters
    ----------
    spec_cat : astropy.table.Table
        Spectroscopic reference catalog.
    phot_path : str
        Path to the photometric FITS catalog.
    save_dir : str
        Directory to write the matched catalog.
    spec_name : str
        Name of the spectroscopic catalog (DESI, JULIA, or ERIK).
    """
    phot_cat = open_fits_catalog(phot_path)
    if spec_name=='DESI':
        matched_catalog = match_cats(phot_cat, spec_cat, sep=0.00027, cat_ref_radec_name=['mean_fiber_ra', 'mean_fiber_dec'])
    else:
        matched_catalog = match_cats(phot_cat, spec_cat, sep=0.00027)
    filename=phot_path.split('/')[-1].replace('.fits', f'spec_match_{spec_name}.fits')
    matched_catalog.write(save_dir+filename,
                         format='fits',
                         overwrite=True)


if __name__=="__main__":
    main()