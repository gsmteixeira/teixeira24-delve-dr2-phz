import numpy as np
from astropy.io import fits
from astropy.table import Table

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

