import argparse
import collections
import json
import os
import pandas
import re
import glob
import sys

import numpy as np
import tensorflow as tf
from model import Model
from sklearn import linear_model, metrics

from vocab import Vocab
import helper
"""
This code is used to evaluate a trained Twitter user embedding model
on the task of community membership classification. Embeddings are 
found for memebers of predefined communities and we learn linear
classifiers to separate those embeddings from the embeddings of
random people from our Twitter collection.

Evaluation is done using the leave-one-out strategy. This means that
in each evaluation run there is one true person belonging to the 
community and a few thousand randos that don't belong to any community.
There are two metrics: AUC and 1/MRR.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, help='experiment directory',
                    default='../models/w2v_init')
parser.add_argument('--datadir', type=str,
                    help='where to find the non-community members')
parser.add_argument('--communities', type=str, 
                    default='../data/communities.csv.gz',
                    help='csv file to load the community tweets from')
args = parser.parse_args()
vocab = Vocab.Load('../data/vocab.txt')

# load the communities
print 'loading communities'
df = pandas.read_csv(args.communities, dtype={'user': str})

# load all the random people
print 'loading randos'
randos = []
filenames = glob.glob(os.path.join(args.datadir, '*.csv'))
for name in filenames:
    randos.append(
        pandas.read_csv(name, dtype={'user':str}, 
                        usecols=['text', 'user', 'timestamp']))
randos = pandas.concat(randos)

# tokenize everything
print 'tokenizing communities'
df = df[~df['text'].isnull()]
df['text'] = df['text'].apply(lambda x: x.split())
randos = randos[~randos['text'].isnull()]

rando_users = randos.groupby('user')
print len(rando_users)

communities = df.groupby('dir')  

# Load the tensorflow embedding model
print 'loading model'
with open(os.path.join(args.expdir, 'params.json'), 'r') as f:
  params = json.load(f)
params['vocab_size'] = len(vocab)
mymodel = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=6,
                        intra_op_parallelism_threads=6)
session = tf.Session(config=config)
saver.restore(session, os.path.join(args.expdir, 'model.bin'))


def Aggregate(users, flag=False, tokenize=False):
  """Helper function to convert tweets grouped by users to dataframe 
     of embeddings."""
  aggregate = []
  for user, df in users:
    df = df.sort_values('timestamp', ascending=False)
    eval_list = df[:50]
    word_ids, counts = helper.GetFeatureVector(eval_list, vocab,
                                               tokenize=tokenize)
    if len(word_ids) < 2:
      continue
    feat_vec = helper.GetEmbed(mymodel, session, word_ids, counts)
    aggregate.append((user, feat_vec))

  return pandas.DataFrame(aggregate, columns=['user', 'feat'])

rando_feats = Aggregate(rando_users, tokenize=True)
rando_feats['label'] = 'rando'

# Split the randos into train and test
train_rando, test_rando = rando_feats[:1000], rando_feats[1000:]
test_feats = np.array([list(x) for x in test_rando.feat.values])
print 'num randos {0}'.format(len(test_rando))

false_positive_search = [] # save top false positives for analysis
all_communities = []  # save these here for analysis
for community, grp in communities:  # loop over all the communities
  ranks = []
  community_users = grp.groupby('user')
  com_users = Aggregate(community_users, flag=True)
  com_users['label'] = 'special' 
  true_probs = []  # need to save true and false probs for AUC computation
  false_probs = []
  for i in range(len(com_users)):  # do leave-one-out
    rest = pandas.concat([com_users[:i], com_users[i+1:]])
    data = pandas.concat([train_rando, rest])
    feats = np.array([list(x) for x in data.feat.values])
    logreg = linear_model.LogisticRegression(random_state=666)
    logreg.fit(feats, data.label)  # train the classifier

    probs = logreg.predict_proba(test_feats)[:, 0]  # predictions for the randos
    true_prob = logreg.predict_proba(com_users.iloc[i].feat.reshape(1, -1))[0][0]
    true_probs.append(true_prob)
    false_probs += list(probs)
    rank = (true_prob > probs).sum() + 1  # rank of true user among randos
    ranks.append(rank)

    for j in np.argsort(probs)[:10]:  # get links to the top 10 randos
      user_id = test_rando.iloc[j].user
      false_positive_search.append({
          'link': 'https://twitter.com/intent/user?user_id={0}'.format(user_id),
          'rank': j,
          'iter': i,
          'prob': 1.0 - probs[j],
          'community': os.path.basename(community)
          })
          
  mrr = (1.0 / np.array(ranks)).mean()  # mean reciprocal rank
  imrr = 1.0 / mrr  # inverse mean recipriocal rank

  # compute the AUC
  labels = ([1] * len(true_probs)) + ([0] * len(false_probs))
  fpr, tpr, _ = metrics.roc_curve(labels, -np.array(true_probs + false_probs))
  auc = metrics.auc(fpr, tpr)
  print '{2}:\t{0:.3f}\t{1:.1f}'.format(imrr, 100.0 * auc, os.path.basename(community))


false_pos_df = pandas.DataFrame(false_positive_search)
z = false_pos_df.groupby(['community', 'link']).agg({'prob': ['mean', len]})
z.to_csv(os.path.join(args.expdir, 'false_positive_report.csv'))
