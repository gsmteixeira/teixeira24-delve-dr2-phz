import os
import sys

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