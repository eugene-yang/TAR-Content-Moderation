import numpy as np
import pandas as pd
import scipy.sparse as sp

import argparse
import pickle
import gzip
import json
from pathlib import Path

import re

from functools import partial
from multiprocessing import Pool

from tqdm import tqdm
from utils import timeit, _document_frequency

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler

from libact.query_strategies import UncertaintySampling, RandomSampling, \
                                    RelevanceFeedbackSampling

from external_classifier_wrapper import VowpalWabbit, SubprocessModelWrapper

RDM_SEED = None
np.random.seed(RDM_SEED)

_strategies = {
    'uncertainty': UncertaintySampling,
    'relevance': RelevanceFeedbackSampling,
    'random': RandomSampling
}

_models = {
    'vworg': partial(VowpalWabbit, training_args={'sgd': None, 
                                                  'adaptive': None, 
                                                  'normalized': None, 
                                                  'invariant': None, 
                                                  'l2': 0.0, 
                                                  'learning_rate': 2}),
    'vw': partial(VowpalWabbit, training_args={'sgd': None, 
                                               'learning_rate': 0.5}),
}

def parsearg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('exp_tag', type=str, 
                        help="Experiment tag of this run.")


    parser.add_argument('--topic_selection', type=str, default='*', 
                        help="Selection of topics seperated by commons. "
                             "This allow `*` and `?` wildcards. ")

    parser.add_argument('--strategies', nargs='+',
                                        choices=_strategies.keys(),
                                        default=['uncertainty', 'relevance'], 
                        help="Choices of sampling strategies. Each sampling "
                             "yields a single experiment run.")
    parser.add_argument('--seed_size', nargs="+", type=int, default=[2], 
                        help="Size of seed set.")
    parser.add_argument('--seed_pos_ratio', nargs="+", type=float, default=[0.5], 
                        help="Ratio of positive examples in the seed set.")

    parser.add_argument('--n_iter', type=int, default=50, 
                        help="Number of TAR iterations.")
    parser.add_argument('--batch_size', type=int, default=100, 
                        help="Batch size of the TAR reviewing.")

    parser.add_argument('--qs_model', choices=_models.keys(), default='vw', 
                        help="The classification model sampling strategy uses.")
    parser.add_argument('--clf_model', choices=_models.keys(), default='vw', 
                        help="The classificaion model for prediction.")
    parser.add_argument('--separate_model', action='store_true', default=False, 
                        help="Whether using the same model for sampling "
                             "strategy and prediction.")

    parser.add_argument('--vw_loss_function', choices=['logistic', 'hinge'], 
                        default='logistic',
                        help="Loss function used by VW.")
    parser.add_argument('--vw_passes', type=int, default=1,
                        help="Number of epoch VW")
    parser.add_argument('--vw_bfgs', action='store_true', default=False, 
                        help="Use bfgs optimizer in VW.")
    parser.add_argument('--vw_absolute_penalty', action='store_true', 
                        default=False, 
                        help="Use absolute penalty in VW.")
    parser.add_argument('--vw_sqrt_penalty', action='store_true', default=False, 
                        help="Use squared root penalty in VW.")

    parser.add_argument('--prior_penalty_style', choices=['l2', 'l1'], 
                        default='l2', help="Regularization penalty.")
    parser.add_argument('--prior_penalty_base', type=float, default=1.0, 
                        help="Regularization penalty strength.")

    parser.add_argument('--random_seed', type=int, default=None, 
                        help="Random seed")

    parser.add_argument('--dataset_path', type=str, default='./wiki_data', 
                        help="Path to ingested dataset.")
    parser.add_argument('--X_file', type=str, default='X_file.npz', 
                        help="Filename of the vectorized collection.")
    parser.add_argument('--output_path', type=str, default='./results', 
                        help="Experiment output directory.")
    parser.add_argument('--exp_name', type=str, default='tar_content_moderation', 
                        help="Experiment name.")

    parser.add_argument('--worker', type=int, default=4, 
                        help="Number of multiprocessing worker.")
    parser.add_argument('--resume', action='store_true', default=False, 
                        help="Resume experiments if partial runs exists.")
    parser.add_argument('--timeproc', action='store_true', default=False, 
                        help="Record processing time.")
    parser.add_argument('--debug', action='store_true', default=False, 
                        help="Run debug mode.")

    args = parser.parse_args()
    dspath = Path( args.dataset_path )

    print(args)

    assert len(args.seed_size) == len(args.seed_pos_ratio) or len(args.seed_pos_ratio) == 1
    if len(args.seed_pos_ratio) == 1:
        args.seed_pos_ratio = args.seed_pos_ratio * len(args.seed_size)

    opath = Path( args.output_path ) / args.exp_name / args.exp_tag
    for p in reversed( opath.parents ):
        if not p.exists():
            p.mkdir()

    if not args.resume or not opath.exists():
        opath.mkdir()
    elif opath.exists():
        print("EXP WARNING: %s exists, might overwrite existing runs."%opath)

    np.random.seed( args.random_seed )

    return dspath, opath, args



def create_ds( X, y, ctl_set_size ):
    ctl_idx = np.random.choice( X.shape[0], ctl_set_size, replace=False )
    ctl_mask = np.zeros(X.shape[0], dtype=bool)
    ctl_mask[ ctl_idx ] = True

    return Dataset( X[ ~ctl_mask ], [None]*(X.shape[0] - ctl_set_size) ), \
           Dataset( X[ ctl_mask ], y[ ctl_mask ] ), \
           Dataset( X[ ~ctl_mask ], y[ ~ctl_mask ] ), \
           ctl_idx


# new evaluation, with residual and rank-frozen metrics
def evaluation(gold, score, known_mask):
    df = pd.DataFrame({ "gold": gold, "score": score, "known": known_mask })
    df = df.assign(adjscore = ((df.gold - 0.5)*np.inf*df.known).fillna(0) + df.score ).sort_values('adjscore', ascending=False)
    df = df.assign(unknown_pos = df.gold & ~df.known)
    n_pos = df.gold.sum()
    rn_pos = df[~df.known].gold.sum()

    dfr80 = np.where((df.gold.cumsum() / n_pos) >= 0.8)[0][0]
    dfr90 = np.where((df.gold.cumsum() / n_pos) >= 0.9)[0][0]

    ndoc = df.shape[0]

    return {
        "P@5": df.gold[:5].sum() / 5,
        "P@20": df.gold[:20].sum() / 20,
        "P@50": df.gold[:50].sum() / 50,
        "R-P": df.gold[:n_pos].sum() / n_pos,

        "rP@5": df[~df.known].gold[:5].sum() / 5,
        "rP@20": df[~df.known].gold[:20].sum() / 20,
        "rP@50": df[~df.known].gold[:50].sum() / 50,
        "rR-P": df.gold[:rn_pos].sum() / rn_pos,
        
        "DFR@0.8": dfr80 / ndoc,
        "DFR@0.9": dfr90 / ndoc,

        # fixed performance
        "AddCost@0.8": (~df.iloc[:dfr80].known).sum(),
        "AddCost@0.8_pos": df.iloc[:dfr80].unknown_pos.sum(),
        "AddCost@0.9": (~df.iloc[:dfr90].known).sum(),
        "AddCost@0.9_pos": df.iloc[:dfr90].unknown_pos.sum(),

        # fixed review cost
        "Fixed20p_pos": df.iloc[: max(0, int(ndoc*0.2 - df.known.sum())) ].unknown_pos.sum(),
        "Fixed40p_pos": df.iloc[: max(0, int(ndoc*0.4 - df.known.sum())) ].unknown_pos.sum(),
        "Fixed6000_pos": df.iloc[: max(0, 6000 - df.known.sum()) ].unknown_pos.sum()

    }


def work( opath, X, niter, batchsize, 
          vw_absolute_penalty, 
          same_model, qs_model, clf_model, 
          separate_model, timeproc, info ):
    tag, y, seedset_idx, strategy = info

    trn_ds = Dataset( X, [None]*X.shape[0] )
    lbr = IdealLabeler( Dataset(X, y) )
    # trn_ds.update( seedset_idx, lbr.label_by_id(seedset_idx) )

    mets = []
    if separate_model or not same_model:
        retraining = True
        qs = strategy(trn_ds, model=qs_model())
        clf = clf_model()
    else: # using the same instance
        retraining = False
        clf = clf_model()
        qs = strategy(trn_ds, model=clf)

    qsname = qs.__class__.__name__
    print(qsname)

    print( tag )
    for i in range(niter+1):
        with timeit(timeproc, "%d all"%i):
            with timeit(timeproc, "%d query"%i):
                if i == 0:
                    ask_id = seedset_idx
                else:
                    if isinstance(qs.model, VowpalWabbit):
                        ask_id = qs.make_query( batchsize, retrain=retraining, absolute_alpha=vw_absolute_penalty )
                    else:
                        ask_id = qs.make_query( batchsize, retrain=retraining )

            lb = lbr.label_by_id( ask_id )
            trn_ds.update(ask_id, lb)

            training_args = {}
            # if isinstance(clf, VowpalWabbit):
            #     training_args['passes_mode'] = 'native'
            # if clf.additional_note is not None and 'total_update' in clf.additional_note:
            #     training_args['passes'] = clf.additional_note['total_update'] // trn_ds.len_labeled() + 1
            #     training_args['passes_mode'] = 'stream'

            with timeit(timeproc, "%d training"%i):
                if isinstance(clf, VowpalWabbit):
                    clf.train(trn_ds, absolute_alpha=vw_absolute_penalty,
                                      **training_args)
                else:
                    clf.train(trn_ds, **training_args)

            # evaluate on full collection
            with timeit(timeproc, "%d scoring"%i):
                score = clf.predict_real( X )[ :, 1 ]

            mets.append( evaluation(y, score, trn_ds.get_labeled_mask() ) )
            # print( mets[-1] )

            # saving
            with timeit(timeproc, "%d saving"%i):
                if clf.__class__.__name__ in ['LogisticRegression']:
                    clf.model.sparsify()
                elif isinstance( clf, SubprocessModelWrapper ):
                    clf.sparsify()
                pickle.dump( clf, gzip.open( opath / ("{}_{}_clf.pkl.gz".format(tag, i)), "w") )
                pickle.dump( {
                    "asked_pos": y[ask_id].sum(),
                    "asked_neg": (~y[ask_id]).sum(),
                    "asked_idx": ask_id,
                    "score": score
                }, gzip.open( opath / ("{}_{}_info.pkl.gz".format(tag, i)), "w") )
    
    
    pickle.dump(mets, gzip.open( opath / ("{}_metrics.pkl.gz".format(tag)), "w"))

if __name__ == '__main__':
    dspath, opath, args = parsearg()

    X = sp.load_npz( dspath / args.X_file )
    rel_info = pd.read_pickle( dspath / "rel_info.pkl" )

    all_topics = rel_info.columns[ ~rel_info.columns.str.contains("meta") ].tolist()
    use_topics = []

    for s in args.topic_selection.split(","):
        if "-" in s:
            use_topics += all_topics[ all_topics.index( s.split("-")[0] ): all_topics.index( s.split("-")[1] )+1 ]
        else:
            pa = re.compile( s.replace("*", ".+").replace("?", ".") )
            use_topics += [ t for t in all_topics if pa.match(t) ]
    use_topics = list(set(use_topics))

    # create seed setsx
    trainsets = { tp:{} for tp in use_topics }
    md5sorted_rel_info = rel_info.sort_values('meta_md5')
    for tp in use_topics:
        posl = md5sorted_rel_info.index[ md5sorted_rel_info[tp] ].tolist()
        negl = md5sorted_rel_info.index[ ~md5sorted_rel_info[tp] ].tolist()
        for i,(sz, posrt) in enumerate( zip(args.seed_size, args.seed_pos_ratio) ):
            p = int(sz * posrt)
            n = sz - p
            trainsets[tp][ f"{tp}_{i}_{p}-{n}" ] = posl[:p] + negl[:n]
            posl = posl[p:]
            negl = negl[n:]
    

    # bind model
    ms = []
    for model_spec in [ args.qs_model, args.clf_model ]:
        t = _models[model_spec]()
        vwargs = t.training_args.copy()
        vwargs.update({ 'passes': args.vw_passes,
                        'loss_function': args.vw_loss_function } )
        if args.vw_bfgs:
            vwargs['bfgs'] = None
        ms.append( partial(VowpalWabbit, penalty=args.prior_penalty_style, alpha=args.prior_penalty_base,
                                         training_args=vwargs) )

    qs_model, clf_model = ms
    same_model = args.qs_model == args.clf_model

    # bind strategy
    stgs = []
    for s in args.strategies:
        if s == 'uncertainty':
            stgs.append( (s, partial(UncertaintySampling, method='lc')) )
        else:
            stgs.append( (s, _strategies[s]) )

    json.dump( vars(args), (opath / "args.json" ).open("w") )

    work_ = partial( work, opath, X, args.n_iter, args.batch_size, 
                     args.vw_absolute_penalty, 
                     same_model,
                     qs_model, clf_model, args.separate_model, args.timeproc )

    # multiprocessing
    print("Multiprocessing with %d workers..."%(args.worker) )
    if args.debug:
        list(map(work_, [
            ("%s_%s"%(tsk, stg), rel_info[tp], tsidx, stgcls)
            for tp in use_topics for tsk, tsidx in trainsets[tp].items() for stg, stgcls in stgs
        ] ))
    else:
        with Pool( args.worker ) as pool:
            pool.map(work_, [
                ("%s_%s"%(tsk, stg), rel_info[tp], tsidx, stgcls)
                for tp in use_topics for tsk, tsidx in trainsets[tp].items() for stg, stgcls in stgs
                if not (opath / ( "%s_%s_metrics.pkl.gz"%(tsk, stg) )).exists()
            ] )





