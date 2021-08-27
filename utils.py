from contextlib import contextmanager
from time import time

@contextmanager
def timeit( output=False, name = None, storage=None):
    s = time()
    yield
    if output:
        print( "" if name is None else (name + ":"), time()-s, "sec" )
    if storage is not None:
        storage.append( time()-s )

import numpy as np
import scipy.sparse as sp

def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
    
import string
def randomString(stringLength=20):
    """Generate a random string of fixed length """
    letters = list(string.ascii_lowercase)
    return "".join(np.random.choice( letters, stringLength ))

    