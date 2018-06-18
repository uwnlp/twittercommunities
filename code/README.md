This directory has code for reproducing the experiments in the community retrieval paper. The first part of that is to learn the word embeddings used for the user representations via the person re-identification task. The word embeddings are learned using the `embedding.py` script.

`python embedding.py --expdir ../models/myexpdir --data ../data/train/1.csv`

The same script can be used to evaluate the person re-identification task.

`python embedding.pyu --mode=eval --expdir ../models/myexpdir --data ../data/val/1.csv`

The person re-identification word embeddings can be used for the community retrieval task.

`python evaluator.py --expdir ../models/myexpdir --communities ../data/communities.csv`

All of the csv data files are created from the raw json dumps obtained from the Twitter API. The `make_dataframe.py` script has the code to convert the json files into the csv input files.
