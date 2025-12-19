"""
Author: Gabriel Teixeira

Utility functions for catalog processing basic operations.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd

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

def match_cats(cat_1, cat_ref, sep=0.00027,
                     cat_1_radec_name=["RA","DEC"], cat_ref_radec_name=["RA","DEC"]):    
    """
    Match the cat_1 catalog with the correspondent cat_ref catalog.

    Parameters:
    cat_1 (fits Table): photometry catalog.
    cat_ref (fits Table): reference catalog.
    cat_1_radec_name: (list): RA and DEC column names for the photometry catalogue
    cat_ref_radec_name: (list): RA and DEC column names for the reference catalogue
    
    Returns:
    fits Table: Matched catalog. Right part: cat_1 features. Second part: cat_ref features. 
    """
    
    
    x = np.array(cat_1[cat_1_radec_name[0]]).astype('float')
    y = np.array(cat_1[cat_1_radec_name[1]]).astype('float')
    
    x_ref = np.array(cat_ref[cat_ref_radec_name[0]]).astype('float')
    y_ref = np.array(cat_ref[cat_ref_radec_name[1]]).astype('float')
    
    cat_1_coords = SkyCoord(ra=x*u.degree, dec=y*u.degree)

    cat_ref_coords = SkyCoord(ra=x_ref*u.degree, dec=y_ref*u.degree) 
    
    idx, d2d, _ = cat_1_coords.match_to_catalog_sky(cat_ref_coords)
    
    
    '''
    idx contem os indices no cat_ref que correspondem a cada um dos objetos no cat1
    depois disso pegamos quais desses indices correspondem a objetos com separação radial menor que 0.00027
    e então indexamos isso ao cat1
    '''
    
    cat_1_matched_mask = np.array(d2d)<=sep # bool.posições no cat1 correspondentes aos objetos que tiveram match 
    cat_ref_matched_idxs = idx[np.array(d2d)<=sep]

    cat_1_match = cat_1.copy()[cat_1_matched_mask]
    cat_ref_cat_matched = cat_ref.copy()[cat_ref_matched_idxs]
    
    
    matched_cat = hstack([cat_1_match, cat_ref_cat_matched])
    
    return matched_cat


def prepare_data(train_path, test_path, zcol_name='Z', dataset=None, scaler=None):
    """
    Load, select, and preprocess photometric data for training and testing.

    This function:
    - Reads training and test catalogs from CSV files
    - Selects a fixed set of magnitude and color features (GRIZ by default)
    - Applies standardization using a provided or newly created scaler
    - Returns feature matrices and target vectors ready for model input

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the test CSV file.
    zcol_name : str
        Name of the redshift column.
    dataset : optional
        Placeholder for dataset selection (currently unused).
    scaler : sklearn scaler, optional
        Scaler instance to apply to the data. If None, a StandardScaler is created.

    Returns
    -------
    x_train : ndarray
        Scaled training features.
    y_train : ndarray
        Training redshift targets.
    x_test : ndarray
        Scaled test features.
    y_test : ndarray
        Test redshift targets.
    scaler : sklearn scaler
        Fitted scaler used in the preprocessing.
    """
    
    if not scaler:
        scaler = StandardScaler()
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # GRIZ
    keys = ['MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z',
            'MAG_G-MAG_R', 'MAG_G-MAG_I', 'MAG_G-MAG_Z',
            'MAG_R-MAG_I', 'MAG_R-MAG_Z','MAG_I-MAG_Z']
    
    #GRI
    # keys = ['MAG_G', 'MAG_R', 'MAG_I',
    #         'MAG_G-MAG_R', 'MAG_G-MAG_I', 'MAG_R-MAG_I',]
    #GRZ
    # keys = ['MAG_G', 'MAG_R', 'MAG_Z',
    #         'MAG_G-MAG_R', 'MAG_G-MAG_Z', 'MAG_R-MAG_Z',]
    keys = [k.replace('MAG_', 'MAG_AUTO_') for k in keys]
    print(keys)
    y_train = train_data[zcol_name].values
    y_test = test_data[zcol_name].values
    x_train = train_data[keys].values
    x_test = test_data[keys].values

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)   

    print(f'X Train Shape = {x_train.shape}')
    print(f'Y Train Shape = {y_train.shape}')
    print(f'X Test Shape = {x_test.shape}')
    print(f'Y Test Shape = {y_test.shape}')

    return x_train, y_train, x_test, y_test, scaler