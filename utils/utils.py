import os
import sys
import numpy as np
def mkdir(directory_path):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory_path : str
        Path to the directory.

    Returns
    -------
    str
        Directory path.
    """
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile
            return sys.exit("Erro ao criar diretório")
        return directory_path

def generate_errors68(pdf, zaxis):
    """
    Compute the 68% confidence interval error from a photometric redshift PDF.

    This function estimates the half-width of the central 68% credible interval
    by integrating the PDF to obtain the cumulative distribution function (CDF)
    and extracting the 16th and 84th percentiles.

    Parameters
    ----------
    pdf : array-like
        One-dimensional probability density function evaluated on `zaxis`.
    zaxis : array-like
        Redshift grid corresponding to the PDF.

    Returns
    -------
    err_68 : float
        Half-width of the 68% confidence interval.
    """
    
    zwidth = zaxis[1]-zaxis[0]
    cdfs = np.cumsum(pdf)*zwidth
    # zoffset=0.01
    p16 = np.sum(cdfs<.1585)*zwidth#np.percentile(pdfs, 15.85, axis=1)
    p84 = np.sum(cdfs<.8405)*zwidth#np.percentile(pdfs, 84.05, axis=1)
    
    err_68 = 0.5*(p84-p16)
    
    return err_68


def shuffle_idx(arr):
    """
    Generate a shuffled index array for a given input array.

    Parameters
    ----------
    arr : array-like
        Input array to be shuffled.

    Returns
    -------
    shuffle_idx : ndarray
        Random permutation of indices with the same length as `arr`.
    """
    
    shuffle_idx = np.random.choice(len(arr), len(arr), replace=False)
    
    return shuffle_idx


def get_z_percentile(pdfs, zaxis, p):
    """
    Compute a percentile redshift from one or more photometric redshift PDFs.

    This function integrates the PDF(s) to form cumulative distribution functions
    and returns the redshift value corresponding to a given percentile.

    Parameters
    ----------
    pdfs : array-like
        One- or two-dimensional array of PDFs.
        If 2D, shape is (n_objects, n_z).
    zaxis : array-like
        Redshift grid corresponding to the PDFs.
    p : float
        Desired percentile in the range [0, 1].

    Returns
    -------
    zperc : float or ndarray
        Redshift percentile(s) corresponding to `p`.
    """

    if np.ndim(pdfs) == 2:
        zwidth = zaxis[1]-zaxis[0]
        cdfs = np.cumsum(pdfs, axis=1)*zwidth

        zperc = np.sum(cdfs<p, axis=1)*zwidth#np.percentile(pdfs, 15.85, axis=1)
    
    else:
        zwidth = zaxis[1]-zaxis[0]
        cdfs = np.cumsum(pdfs)*zwidth

        zperc = np.sum(cdfs<p)*zwidth#np.percentile(pdfs, 15.85, axis=1)
     
        
    return zperc