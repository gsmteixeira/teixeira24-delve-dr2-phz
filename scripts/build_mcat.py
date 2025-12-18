import os
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.utils import mkdir
from utils.processing import open_fits_catalog, modest_class_mask

def main():
    """
    Main entry point for cleaning DELVE DR2 photometric catalogs.

    Applies magnitude, flag, and MODEST star–galaxy classification
    cuts to each catalog and saves the cleaned outputs.
    """
    #DELVE DR2 data available at https://datalab.noirlab.edu/data-explorer?showTable=delve_dr2.objects 
    DELVE_DR2_PATH = '/tf/astrodados/DELVE_DR2_cleaned/' # data/delve_dr2/
    DELVE_DR2_FILES = [os.path.join(DELVE_DR2_PATH, file) for file in os.listdir(DELVE_DR2_PATH) if file.endswith('.fits')] 

    SAVE_DIR =  mkdir('data/mcat/')
    NUM_PROCESS = 6

    with open(SAVE_DIR + 'delve_cleaning_summary.txt', 'w') as f:
        # f.write(f'')
        f.close()

    start_clean = time.time()
    Parallel(n_jobs=NUM_PROCESS)(delayed(clean_catalog)\
                                (file_path=DELVE_DR2_FILES[i], save_dir=SAVE_DIR)\
                                for i in tqdm(range(1))) # for the complete analysis use tqdm(range(len(DELVE_DR2_FILES)))
    end_clean = time.time()

    with open(SAVE_DIR + 'delve_cleaning_summary.txt', 'a') as f:
        f.write(f'ELAPSED TIME = {(end_clean - start_clean)/60:.2f} min\n')
        f.close()

def mag_g_mask(cat_, maglim=23.5,g_key='MAG_AUTO_G'): 
    """
    Create a boolean mask selecting objects brighter than a given g-band magnitude.
    """
    mask = np.array(cat_[g_key])<maglim
    
    return mask

def flag_g_mask(cat_, flag_key='FLAGS_G', flaglim=3): 
    """
    Create a boolean mask selecting objects with acceptable g-band quality flags.
    """
    mask = np.array(cat_[flag_key])<flaglim
    
    return mask

def clean_catalog(file_path, save_dir):
    """
    Clean a single DELVE DR2 catalog.

    Applies magnitude, flag, and MODEST classification cuts,
    writes the cleaned catalog to disk, and logs summary statistics.
    """
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

if __name__=="__main__":
    main()