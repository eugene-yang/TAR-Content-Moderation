# TAR on Social Media: A Framework for Online Content Moderation

This repository contains experiment code and framework of TAR reviewing process 
that facilitates continue human-in-the-loop content moderation. 

## Installation

### Dependencies
[VowpalWabbit](https://vowpalwabbit.org/index.html) is not included in the conda environment. Please refer to their installation guide for installing VwolpalWabbit to command line. 

The following command creates and activates the conda environment for the experiment.

```bash
LIBACT_BUILD_VARIANCE_REDUCTION=0 conda env create --file environment.yml 
conda activate tarcm
```

## Ingest dataset
Both Wikipedia Personal Attack dataset and AskFM Cyberbullying  dataset need to be ingested before the experiments. This step parses the annotations and raw text to an unified format. 

### Wikipedia Personal Attack
- Download dataset: https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689
- Put these three files into a directory
- Run `python ingest_wiki.py --dataset {raw dataset directory} --output {output directory}`. If needed, a flag `--overwrite` can be added for overwriting an existing output directory. 

### AskFM Cyberbullying Dataset
- Please contact the authors of paper [Automatic detection of cyberbullying in social media text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203794) for data access. 
- Uncompress the dataset into a directory
- Run `python ingest_askfm.py --dataset {raw dataset directory} --output {output directory}`. If needed, a flag `--overwrite` can be added for overwriting an existing output directory. 

## Experiments

Detailed arguments is provided in `python tar_exp.py --help`.

The following is an example of running 80 rounds of TAR review with a batch size
of 100 on the Wikipedia Personal Attack dataset. 

```bash
python tar_exp.py wiki-100x80 \
                  --topic_selection * --strategies uncertainty relevance \
                  --seed_size 2 --see_pos_ratio=0.5
                  --n_iter 80 --batch_size 100 \
                  --vw_loss_function logistic --vw_passes 1 \
                  --random_seed 123 \
                  --dataset_path ./wiki_ingested_data/ \
                  --X_file X_file.npz \
                  --output_path ./results \
                  --exp_name tar_content_moderation \
                  --worker 4
```

## Reference
If you adapt scripts or code from this repository or find this related to your 
work, please kindly cite our paper. 

```
@inproceedings{tar_on_social_media,
	title={TAR on Social Media: A Framework for Online Content Moderation},
	author={E. Yang and D. D. Lewis and O. Frieder},
	journal = {Proceddings of Design of Experimental Search & Information REtrieval Systems 2021 (DESIRES 2021)},
	year={2021}
}
```
