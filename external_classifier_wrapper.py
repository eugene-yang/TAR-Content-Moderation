"""
This module includes a class for interfacing VW classifier via subprocess call.
"""
import numpy as np
import scipy.sparse as sp
import subprocess

import shutil, os

import tempfile

from libact.base.interfaces import ProbabilisticModel
from abc import abstractmethod

from utils import timeit
from pathlib import Path

def _cache_key(X, args=[], kwargs={}):
    if isinstance( X, np.ndarray ):
        X = X.tobytes()
    elif isinstance( X, sp.spmatrix ):
        nz = X.nonzero()
        X = tuple([nz[0].tobytes(), nz[1].tobytes , X.data.tobytes])

    kk = tuple(sorted(kwargs.keys()))
    kv = tuple([ kwargs[k] for k in kk ])
    return tuple([ X, tuple(args), kk, kv ])

class PredictionCache(object):
    def __init__(self, size=10, style='LRU'):
        self._size = size
        self._storage = {}
        self._priority = []
        if style not in ['LRU', 'FIFO']:
            raise ValueError("Style should be either `LRU` or `FIFO`")
        self._style = style

    @property
    def capacity(self): return self._size

    @property
    def usage(self): return len(self._priority)

    def __len__(self): return self.usage

    def __contains__(self, key):
        return _cache_key(*key) in self._priority

    def clear(self):
        self._storage = {}
        self._priority = []

    def __setitem__(self, key, value):
        key = _cache_key( *key )
        if key in self._priority:
            self._priority.remove(key)
            del self._storage[key]
        if len(self._priority) == self._size:
            rk = self._priority.pop()
            del self._storage[rk]
        self._priority.append( key )
        self._storage[ key ] = value

    def __getitem__(self, key):
        key = _cache_key( *key )
        if key not in self._priority:
            raise KeyError
        if self._style == 'LRU':
            self._priority.remove(key)
            self._priority.insert(0, key)
        return self._storage[ key ]

def _same_matrix( a, b ):
    try:
        if a.__class__ != b.__class__:
            return False
        elif a is None and b is None:
            return True
        elif isinstance( a!=b, bool ):
            return a == b
        elif isinstance( a, sp.spmatrix ):
            return (a != b).data.size == 0
        return not ( a != b ).any()
    except Exception as e:
        print(a.__class__, b.__class__)
        print( a!=b )
        raise e


class SubprocessModelWrapper(ProbabilisticModel):
    """SubprocessModelWrapper

    An abstract wrapper around classifiers called via suprocess calls.
    Only supports binary classification models.

    """

    def __init__(self, cache_size=10, *args, **kwargs):
        # setup cache
        self._last_trained_on = None, None
        self._predict_cache = PredictionCache(cache_size)
        self.additional_note = None

        self.coef_ = None
        self.intercept_ = 0

    def clear_cache(self):
        self._predict_cache.clear()

    def train(self, dataset, force=False, *args, **kwargs):

        X, y = dataset.get_labeled_entries()
        # if not force and \
        #    _same_matrix(X, self._last_trained_on[0]) and ( np.array(y) == self._last_trained_on[1] ).all():
        #    # retraining with the same training set,
        #    return False

        # actual training
        # self._last_trained_on = X, y
        # self.clear_cache()
        # self._call_train( *self._last_trained_on, *args, **kwargs )
        self._call_train( X, y, *args, **kwargs )
        return True

    def trained(self): return self.coef_ is not None

    def decision_function(self, feature):
        assert self.trained()
        score = feature * self.coef_.T
        if isinstance( score, sp.spmatrix ):
            score = score.todense()
        if score.shape[1] == 1:
            return score.A1 + self.intercept_
        else:
            return score + self.intercept_

    def predict(self, feature, *args, **kwargs):
        return self.predict_proba(feature)[:,1] > 0

    def score(self, testing_dataset, *args, **kwargs):
        X, y = testing_dataset.get_labeled_entries()
        return ( self.predict(X) == np.array(y) ).sum() / len(y)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.decision_function( feature, *args, **kwargs )
        return np.vstack((-dvalue, dvalue)).T

    def predict_proba(self, feature, *args, **kwargs):
        # logit function
        return 1 / ( 1 + np.exp(-self.predict_real( feature, *args, **kwargs )) )

    def sparsify(self):
        self.coef_ = sp.csr_matrix(self.coef_)

    @abstractmethod
    def _call_train(self, X, y, *args, **kwargs):
        pass


vw_default_args = {
    'cache': None,
    'kill_cache': None,
    'loss_function': 'logistic',
    'noconstant': None,
    "holdout_off": None,
    'passes': 1,
    'sgd': None, # IMPORTANT!!: Force VW to get the 'right' answer
    'learning_rate': 0.5,
    'bit_precision': 25
}

class VowpalWabbit(SubprocessModelWrapper):

    """Vowpal Wabbit Classifier

    References
    ----------
    """

    def __init__(self, penalty='l2', alpha=1.0, *args, **kwargs):
        super( VowpalWabbit, self ).__init__( *args, **kwargs )
        self.training_args = vw_default_args.copy()

        assert penalty in ['l1', 'l2']

        self.training_args.update( kwargs['training_args'] if 'training_args' in kwargs else {} )
        self.training_args[ penalty ] = alpha

        if 'additional_note' in kwargs:
            self.additional_note = kwargs['additional_note']

        self._vw_path = 'vw' if shutil.which('vw') is not None else os.environ['VW']

    @staticmethod
    def encode_vw_data(X, y, sample_weights=None, intercept_weight=1.0):
        if sample_weights is None:
            sample_weights = np.ones(len(y))

        assert len(y) == len(sample_weights)

        return "\n".join([
            ( "{} {} | {}:{} ".format( 1 if c else -1, w, X.shape[1] + 1000, intercept_weight ) ) +
            " ".join([ "{}:{:.8f}".format( k[-1], v[k] ) for k in zip(*v.nonzero()) ])
            for v,c,w in zip( X,y,sample_weights )
        ])

    @staticmethod
    def read_vw_coef( fn, length ):
        lines = open( fn ).read().split("\n")[:-1]

        intercept = 0
        idx, values = [], []
        for l in lines[11:]:
            l = l.split(":")
            if int(l[0]) > length-1: # intercept
                intercept = float(l[1])
            else:
                idx.append( int(l[0]) )
                values.append( float(l[1]) )

        return intercept, sp.csr_matrix(( values, ( [0]*len(idx), idx ) ), shape=(1, length)).todense()


    def _call_train(self, X, y, sample_weights=None,
                    intercept_weight=1.0,
                    prior_strength=None, prior_intercept=(None, None),
                    passes_mode="native", manual_file=None,
                    absolute_alpha=False, sqrt_alpha=False, *args, **kwargs):
        # internal testing purpose
        given_file = False
        if isinstance( manual_file, str ):
            # passing in the filename directly
            fX_name = manual_file
            given_file = True
        else:
            assert X.shape[0] == len(y)

        vwargs = self.training_args.copy()
        vwargs.update( kwargs )

        if 'bfgs' not in vwargs and 'conjugate_gradient' not in vwargs and\
            not absolute_alpha and len(y) > 0:
            if 'l2' in vwargs:
                vwargs['l2'] = vwargs['l2'] / len(y)
            if 'l1' in vwargs:
                vwargs['l1'] = vwargs['l1'] / len(y)

        if sqrt_alpha:
            assert not absolute_alpha
            if 'l2' in vwargs:
                vwargs['l2'] = vwargs['l2'] / np.sqrt( len(y) )
            if 'l1' in vwargs:
                vwargs['l1'] = vwargs['l1'] / np.sqrt( len(y) )

        if prior_strength is not None or prior_intercept != (None, None):
            raise NotImplementedError("prior_strength and prior_intercept is not "
                                      "supported in VowpalWabbit.")

        with tempfile.NamedTemporaryFile("w") as fX, \
             tempfile.NamedTemporaryFile() as fmodel, \
             tempfile.NamedTemporaryFile() as fintm:

            if passes_mode == "native":
                if 'passes' in vwargs and vwargs['passes'] > 1:
                    vwargs['cache'] = None
                if not given_file:
                    fX.write( VowpalWabbit.encode_vw_data(X, y, sample_weights, intercept_weight) )
                    fX.flush()
                p = subprocess.run([self._vw_path, *[ "--{}={}".format(k, v) if v is not None else "--{}".format(k)
                                                                            for k,v in vwargs.items() ],
                                                '-d', fX.name if not given_file else fX_name,
                                                '--readable_model', fmodel.name ],
                                    text = True,
                                    stderr = subprocess.PIPE, stdout = subprocess.PIPE)

                if p.returncode != 0:
                    raise RuntimeError(p.stderr)
                try:
                    os.remove( fX.name + ".cache" )
                except:
                    pass

                self.intercept_, self.coef_ = VowpalWabbit.read_vw_coef( fmodel.name, length=X.shape[1] )

            elif passes_mode == "bigfile":
                ps = vwargs['passes']
                vwargs['passes'] = 1

                if not given_file:
                    Xstr = "\n".join([ VowpalWabbit.encode_vw_data(X, y, sample_weights, intercept_weight) ]*ps)
                    fX.write( Xstr )
                    fX.flush()

                p = subprocess.run([self._vw_path, *[ "--{}={}".format(k, v) if v is not None else "--{}".format(k)
                                                                            for k,v in vwargs.items() ],
                                                '-d',  fX.name if not given_file else fX_name,
                                                '--readable_model', fmodel.name ],
                                    text = True,
                                    stderr = subprocess.PIPE, stdout = subprocess.PIPE)

                if p.returncode != 0:
                    raise RuntimeError(p.stderr)

                # print( p.args )

                self.intercept_, self.coef_ = VowpalWabbit.read_vw_coef( fmodel.name, length=X.shape[1] )

            elif passes_mode == 'stream':
                ps = vwargs['passes']
                vwargs['passes'] = 1

                p = subprocess.Popen([self._vw_path, *[ "--{}={}".format(k, v) if v is not None else "--{}".format(k)
                                                                            for k,v in vwargs.items() ],
                                                '--readable_model', fmodel.name ],
                                    text = True,
                                    stdin = subprocess.PIPE,
                                    stderr = subprocess.PIPE, stdout = subprocess.PIPE)

                Xstr = VowpalWabbit.encode_vw_data(X, y, sample_weights, intercept_weight)
                for _ in range(ps):
                    p.stdin.write( Xstr + "\n" )

                if p.returncode is not None:
                    raise RuntimeError(p.stderr.read())

                p.stdin.close()

                # might be something wrong after termination
                if p.returncode != 0:
                    raise RuntimeError("Something wrong after terminate VW.")

                # print( p.args )

                self.intercept_, self.coef_ = VowpalWabbit.read_vw_coef( fmodel.name, length=X.shape[1] )
