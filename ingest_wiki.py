import pandas as pd
from pathlib import Path
from hashlib import md5
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp

import argparse

def main(args):
    if args.output.exists():
        if not args.overwrite():
            raise FileExistsError(f"Output directory {args.output} exists.")
        print(f"Output directory {args.output} exists. It will be overwritten.")
    args.output.mkdir(exist_ok=True, parents=True)

    raw_ds = Path(args.dataset)
    annot = pd.read_csv( raw_ds / 'attack_annotations.tsv', sep='\t' )

    d = annot.groupby('rev_id').mean().drop(['worker_id', 'quoting_attack'], axis=1).pipe(lambda x: x>0.5)
    rel_info = d.rename_axis('meta_rev_id').reset_index().assign(meta_md5=d.index.astype('str').map(lambda x: md5(x.encode()).hexdigest()))

    raw_text = pd.read_csv( raw_ds / "attack_annotated_comments.tsv", sep='\t' )
    assert (raw_text.rev_id == rel_info.meta_rev_id).all()

    X = TfidfVectorizer(sublinear_tf=True, use_idf=False).fit_transform(raw_text.comment)

    print("Saving files...")
    rel_info.to_pickle( args.output / "rel_info.pkl" )
    sp.save_npz(str(args.output / "X_file.npz"), X )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path) # /media/eugene/research/datasets/wiki-personal
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true', default=False)

    main(parser.parse_args())