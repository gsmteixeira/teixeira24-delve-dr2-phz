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

def file2coords(file_path):
    """
    Converts the block name into the vertices (lower left, upper right) coordinates .

    Parameters:
    file_path (str): complete path of the Legacy file 
  
    Returns:
    tuple of lists: each list containing ra and dec coordinates, respectively.
    """
    
    filename = file_path.split('/')[-1]
    coordinates = filename.replace('.fits', '').replace('sweep-', '')
    signal = {'m':-1, 'p':1}
    ra1 = float(coordinates.split('-')[0][:3])
    ra2 = float(coordinates.split('-')[1][:3])
    dec1 = float(coordinates.split('-')[0][4:])*signal[coordinates.split('-')[0][3]]
    dec2 = float(coordinates.split('-')[1][4:])*signal[coordinates.split('-')[1][3]]

    
    
    ra = [ra1, ra2]
    dec = [dec1, dec2]
    
    return ra, dec

def check_lim_dec(file_path):
    """
    Check if the lower limit of the legacy imaging block corresponds to the DELVE DR2 imaging area.

    Parameters:
    file_path (str): complete path of the Legacy file 
  
    Returns:
    bool: True if the block corresponds to DELVE; False if it not corresponds.
    """
    _, dec = file2coords(file_path)
    
    return min(dec) < 35

def hpx2radec(file_path, radius=None, npoints = 100):
    """
    Converts the DELVE healpix name into the healpix coordinates.

    Parameters:
    file_path (str): complete path of the DELVE DR2 file 
    radius (float): size in deg of the radius of the circle entered on the healpix RA and DEC coordinates. 
  
    Returns:
    tuple of lists: lists of points that compose a circle centered on the healpix RA and DEC coordinates, respectively. 
                    If radius==None, returns just the healpix corrdinates.
    
    """
    
    file = file_path.split('/')[-1]#
    hpidx = int(file[file.find('hpx_')+4:file.find('hpx_')+9])    
    theta, phi = hp.pixelfunc.pix2ang(32, hpidx) # pixel encoded in 2^5 = 32 nsize
    ra = np.degrees(phi)
    dec = np.degrees(np.pi/2 - theta)

    if radius:
        R = radius

        param_t = np.linspace(0, np.pi*2, 40)

        bound_ra = np.array([ra]*40)+R*np.cos(param_t)
        bound_dec = np.array([dec]*40)+R*np.sin(param_t)

        bound_dec[bound_dec<-90] = -90
        bound_dec[bound_dec>90] = 90
        return bound_ra, bound_dec
    
    else:
        return ra, dec
    
def check_intersec(delve_file_path, legacy_file_path, radius=11.5):   
    """
    Check if there is intersection between the DELVE imaging and the Legacy block.

    Parameters:
    legacy_file_path (str): complete path of the Legacy file.
    delve_file_path (str): complete path of the DELVE DR2 file. 
  
    Returns:
    bool: NTrue if there is a intersection; False if there is not.
    """
    
    delve_ra, delve_dec = hpx2radec(delve_file_path, radius=radius, npoints = 100)
    legacy_ra, legacy_dec = file2coords(legacy_file_path)
    
    insq = (delve_ra>legacy_ra[0])*(delve_ra<legacy_ra[1])*(delve_dec>legacy_dec[0])*(delve_dec<legacy_dec[1])
    
    return sum(insq)>0

def count_insquare(delve_ra, delve_dec, legacy_file_path):   
    """
    Check if there is intersection between the DELVE imaging and the Legacy block.

    Parameters:
    legacy_file_path (str): complete path of the Legacy file.
    delve_file_path (str): complete path of the DELVE DR2 file. 
  
    Returns:
    bool: NTrue if there is a intersection; False if there is not.
    """
    

    legacy_ra, legacy_dec = file2coords(legacy_file_path)
    
    insq = (delve_ra>legacy_ra[0])*(delve_ra<legacy_ra[1])*(delve_dec>legacy_dec[0])*(delve_dec<legacy_dec[1])
    
    return sum(insq)

def match_delve_legacy(delve_cat, legacy_cat_ref, sep=0.00027,
                     delve_radec_name=["RA","DEC"], legacy_radec_name=["RA","DEC"]):    
    """
    Match the delve catalog with the correspondent legacy catalog.

    Parameters:
    delve_cat (fits Table): DELVE catalog.
    legacy_cat (fits Table): Legacy catalog.
    delve_radec_name: (list): RA and DEC column names for the DELVE catalog
    legacy_radec_name: (list): RA and DEC column names for the Legacy catalog
    
    Returns:
    fits Table: Matched catalog. Right part: DELVE features. Second part: Legacy features. 
    """
    
    
    x = np.array(delve_cat[delve_radec_name[0]]).astype('float')
    y = np.array(delve_cat[delve_radec_name[1]]).astype('float')
    
    x_ref = np.array(legacy_cat_ref[legacy_radec_name[0]]).astype('float')
    y_ref = np.array(legacy_cat_ref[legacy_radec_name[1]]).astype('float')
    
    delve_coords = SkyCoord(ra=x*u.degree, dec=y*u.degree)

    legacy_coords = SkyCoord(ra=x_ref*u.degree, dec=y_ref*u.degree) 
    
    idx, d2d, d3d = delve_coords.match_to_catalog_sky(legacy_coords)
    
    
    '''
    idx contem os indices no cat_ref que correspondem a cada um dos objetos no cat1
    depois disso pegamos quais desses indices correspondem a objetos com separação radial menor que 0.00027
    e então indexamos isso ao cat1
    '''
    
    delve_matched_mask = np.array(d2d)<=sep # bool.posições no cat1 correspondentes aos objetos que tiveram match 
    legacy_matched_idxs = idx[np.array(d2d)<=sep]

    delve_cat_match = delve_cat.copy()[delve_matched_mask]
    legacy_cat_matched = legacy_cat_ref.copy()[legacy_matched_idxs]
    
    
    matched_cat = hstack([delve_cat_match, legacy_cat_matched])
    
    return matched_cat

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

def match_loop(spec_cat, legacy_path, save_dir, spec_name):
    
    legacy_cat = open_fits_catalog(legacy_path)
    if spec_name=='DESI':
        matched_catalog = match_delve_legacy(legacy_cat, spec_cat, sep=0.00027, legacy_radec_name=['mean_fiber_ra', 'mean_fiber_dec'])
    else:
        matched_catalog = match_delve_legacy(legacy_cat, spec_cat, sep=0.00027)
    filename=legacy_path.split('/')[-1].replace('.fits', f'spec_match_{spec_name}.fits')
    # print(f'working on {filename} == ')
    # if matched_catalog:
    
    matched_catalog.write(save_dir+filename,
                         format='fits',
                         overwrite=True)
    
    

    
    
    
# DELVE_DR2_R1_PATH = '/tf/dados10T/delve_dr2_cats/r1/'
# DELVE_DR2_R2_PATH = '/tf/dados10T/delve_dr2_cats/r2/'
DELVE_CLEANED_PATH =  '/tf/astrodados/DELVE_DR2_cleaned/'
DELVE_DR2_FILES = [DELVE_CLEANED_PATH + file for file in os.listdir(DELVE_CLEANED_PATH) if file.endswith('.fits')]


MATCHED_DIR = mkdir('/tf/astrodados/DELVE_DR2_Xmatch_specz/')

DESI_DR1_PATH = '/tf/astrodados/DESi_EDR/'
JULIA_CATALOG_PATH = '/tf/dados10T/specz_matches/'
ERIK_CATALOG_PATH = '/tf/astrodados/erik_catalog/'    
    
DESI_CATALOG_FILES = [DESI_DR1_PATH + 'DESi_EDR.fits']
JULIA_CATALOG_FILES = [JULIA_CATALOG_PATH + 'SPECZ_24NOV20.fits']
ERIK_CATALOG_FILES = [ERIK_CATALOG_PATH + 'SpecZ_Catalogue_20230704.csv']

spec_num_list = []
delve_num_list = []
NUM_PROCESS = 6

delve_names = []
desi_data = open_fits_catalog(DESI_CATALOG_FILES[0])#[:1000]
julia_data = open_fits_catalog(JULIA_CATALOG_FILES[0])#[:1000]
erik_data = Table.from_pandas(pd.read_csv(ERIK_CATALOG_FILES[0])[['RA','DEC','z', 'e_z', 'class_spec']])#[:1000]



# DESI LOOP
start_desi = time.time()
Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(legacy_path=DELVE_DR2_FILES[i], spec_cat=desi_data, save_dir=MATCHED_DIR, spec_name='DESI') for i in tqdm(range(len(DELVE_DR2_FILES))))
end_desi = time.time()    
with open(MATCHED_DIR+'DESI_match_time.txt', 'w') as f:
        f.write(f'elapsed time doing DESI match = {(end_desi-start_desi)/60:.2f} min\n')
        f.close()

# JULIA LOOP
start_julia = time.time()
Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(legacy_path=DELVE_DR2_FILES[i], spec_cat=julia_data, save_dir=MATCHED_DIR, spec_name='JULIA') for i in tqdm(range(len(DELVE_DR2_FILES))))
end_julia = time.time()    
with open(MATCHED_DIR+'julia_match_time.txt', 'w') as f:
        f.write(f'elapsed time doing julia match = {(end_julia-start_julia)/60:.2f} min\n')
        f.close()


# ERIK LOOP
start_erik = time.time()
Parallel(n_jobs=NUM_PROCESS)(delayed(match_loop)(legacy_path=DELVE_DR2_FILES[i], spec_cat=erik_data, save_dir=MATCHED_DIR, spec_name='ERIK') for i in tqdm(range(len(DELVE_DR2_FILES))))
end_erik = time.time()    
with open(MATCHED_DIR+'ERIK_match_time.txt', 'w') as f:
        f.write(f'elapsed time doing ERIK match = {(end_erik-start_erik)/60:.2f} min\n')
        f.close()
