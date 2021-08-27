from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from hashlib import md5
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import sparse as sp

import argparse

def break_text(raw):
    return np.array([ i for i, t in enumerate(raw) if t == '¶' ][::2]) 

def main(args):
    if args.output.exists():
        if not args.overwrite():
            raise FileExistsError(f"Output directory {args.output} exists.")
        print(f"Output directory {args.output} exists. It will be overwritten.")
    args.output.mkdir(exist_ok=True, parents=True)

    ds_path = Path(args.dataset)

    raw_text = {}
    break_idx = {}
    for fn in tqdm(list((ds_path / "en").glob("*.txt")), desc='Parsing text'):
        fid = fn.name.split("_")[2]
        raw = fn.read_text()
        idx = break_text(raw)
        break_idx[ fid ] = np.array(idx)
        for i in range(len(idx)):
            t = raw[idx[i]:] if i == len(idx)-1 else raw[idx[i]:idx[i+1]]
            raw_text[f"{fid}_{i}"] = t.replace('¶', '').strip()
        
    raw_text = pd.Series(raw_text).sort_index()

    rel = {}
    for fn in tqdm(list((ds_path / "en").glob("*.ann")), desc='Parsing annotations'):
        fid = fn.name.split("_")[2]
        for annl in fn.open():
            tp, bidx, eidx = annl.strip().split("\t")[1].split(" ")
            if len(break_idx[fid]) == 1:
                pass_id = 0
            else:
                pass_id = (break_idx[fid] <= int(bidx)).cumsum()[-1]-1
            assert pass_id >= 0
            rel[ f"{fid}_{pass_id}", tp ] = True

    rel_info = pd.Series(rel).sort_index().unstack(1)\
                .join(raw_text.rename('text'), how='right')\
                .drop(['text', 'Other', 'Other_language'], axis=1).fillna(False)

    rel_info = rel_info.rename_axis('meta_pid')\
                .assign(meta_md5=rel_info.index.astype('str').map(lambda x: md5(x.encode()).hexdigest()))\
                .reset_index()

    assert (raw_text.index == rel_info.meta_pid).all()

    X = TfidfVectorizer(sublinear_tf=True, use_idf=False).fit_transform(raw_text)

    print("Saving files...")
    rel_info.to_pickle( args.output / "rel_info.pkl" )
    sp.save_npz( str(args.output / "X_file.npz"), X )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path) # '/media/eugene/research/datasets/askfm-cyberbullying-data'
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true', default=False)

    main(parser.parse_args())