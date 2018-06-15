"""
Module to train or evaluate RE-ID model
"""
import argparse
import random
import os
import gzip
import json
import glob
import logging

import numpy as np
import pandas
import tensorflow as tf
import gensim

import helper
from metrics import MovingAvg
from model import Model
from vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('expdir', type=str, help='experiment directory')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                    help=('use training mode to train and eval mode to evaluate on'
                          ' the proxy task'))
parser.add_argument('--threads', type=int, default=8, help='num threads')
parser.add_argument('--data', type=str, action='append', dest='data',
                    default=None)
parser.add_argument('--params', type=str, default='default_params.json',
                    help='json file with hyperparameter settings')
args = parser.parse_args()

if not os.path.exists(args.expdir):
    os.mkdir(args.expdir)

logging.basicConfig(filename=os.path.join(args.expdir, 'logfile.txt'),
                    level=logging.INFO)
stderrLogger = logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)

with open(args.params, 'r') as f:
    params = json.load(f)

random.seed(666)
vocab = Vocab.Load('/g/ssli/data/surf/w2v_vocab.pickle')

def build_df(data):
    dfs = []
    for filename in data:
        logging.info('loading {0}'.format(filename))
        df = pandas.read_csv(filename, dtype={'user': str})
        print 'tokenizing'
        df = df[~df['text'].isnull()]
        df['text'] = df['text'].apply(lambda x: [w for w in x.split() if w in vocab])
        dfs.append(df)
    df = pandas.concat(dfs)
    groups = df.groupby('user')
    user_ids = df.user.unique()
    return groups, user_ids

params['vocab_size'] = len(vocab)

# save parameters in expdir
with open(os.path.join(args.expdir, 'params.json'), 'w') as f:
    json.dump(params, f)


def GetRandomTriplet(groups, user_ids):
    """
    Get a random triplet of positive tweets, anchor tweets
    and negative tweets. Positive and anchor tweets must come from
    same person, and negative tweets must come from a different person

    Returns postive tweet features, positive tweet word counts,
    anchor tweet features, anchor tweet word counts,
    negative tweet features, negative tweet word counts
    """
    pos_id, neg_id = random.sample(user_ids, 2)
    neg_tweets = groups.get_group(neg_id)
    pos_tweets = groups.get_group(pos_id)

    num_tweets = params['tweets_per_person']
    while len(neg_tweets) < num_tweets or len(pos_tweets) < 2 * num_tweets:
        pos_id, neg_id = random.sample(user_ids, 2)
        neg_tweets = groups.get_group(neg_id)
        pos_tweets = groups.get_group(pos_id)

    pos_sample = pos_tweets.sample(2 * num_tweets)

    neg_feats, neg_counts = helper.GetFeatureVector(neg_tweets.sample(num_tweets), vocab)
    pos_feats, pos_counts = helper.GetFeatureVector(pos_sample[:num_tweets], vocab)
    anchor_feats, anchor_counts = helper.GetFeatureVector(pos_sample[num_tweets:], vocab)
    return pos_feats, pos_counts, anchor_feats, anchor_counts, neg_feats, neg_counts


mymodel = Model(params)  # create the model

saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

if params['use_pretrained_embeddings']:
    logging.info('loading and initializing pre-trained embeddings')
    embeddings = np.zeros((len(vocab), 128))
    with gzip.open('/g/ssli/data/word2vec_full.txt.gz', 'r') as f:
        for line in f:
            fields = line.split()
            word = fields[0]
            index = vocab[word]
            embed = [float(x) for x in fields[1:]]
            embeddings[index, :] = embed
    embed_placeholder = tf.placeholder(tf.float32, [len(vocab), 128], name='embed_placeholder')
    assign_op = tf.assign(mymodel.word_embeddings, embed_placeholder)
    session.run(assign_op, {embed_placeholder: embeddings})

def GetFeedDict(df, user_ids):
    """Pick a random triplet and create the feed dict for it."""
    pos, pos_counts, anchor, anchor_counts, neg, neg_counts = GetRandomTriplet(df, user_ids)
    while min(len(pos), len(anchor), len(neg)) < 10:
        pos, pos_counts, anchor, anchor_counts, neg, neg_counts = GetRandomTriplet(
            df, user_ids) # skip samples with not many words

    feed_dict = {mymodel.word_ids_pos: pos,
                 mymodel.word_ids_neg: neg,
                 mymodel.word_ids_anchor: anchor,
                 mymodel.word_count_pos: pos_counts,
                 mymodel.word_count_neg: neg_counts,
                 mymodel.word_count_anchor: anchor_counts}
    return feed_dict

def Train():
    """Train the model, feeding in a random triplet every iteration."""
    if args.data:
        data = args.data
    else:
        data = glob.glob('/g/ssli/data/surf/csv/train_df_*')
    df, user_ids = build_df(data) 

    avgloss = MovingAvg()  # less noisy to print moving average of loss
    for idx in range(params['iters']):
        feed_dict = GetFeedDict(df, user_ids)
        print feed_dict
        cost, _, d_pos, d_neg = session.run(
            [mymodel.loss, mymodel.train_op, mymodel.d_pos, mymodel.d_neg], feed_dict)
        if np.isnan(cost):
            logging.error('loss is nan')
            exit()  # no point in continuing now

        c = avgloss.Update(cost)
        if idx % 10 == 0:  # print loss every ten timesteps
            logging.info({'iter': idx, 'cost': c, 'd_pos': d_pos, 'd_neg': d_neg})

        if idx % 1000 == 0:  # checkpoint model every 1000 timesteps
            saver.save(session, os.path.join(args.expdir, 'model.bin'))
    saver.save(session, os.path.join(args.expdir, 'model.bin'))


def Eval():
    """This function evaluates on the person re-identification proxy task.

    This is not used for evaluting on the community detection task.
    """
    if args.data:
        data = args.data
    else:
        data = glob.glob('/g/ssli/data/surf/csv/test_df_*')

    df, user_ids = build_df(data) 
    saver.restore(session, os.path.join(args.expdir, 'model.bin'))

    total_cost = 0.0
    for _ in range(1000):  # just evaluate on 1,000 random triplets
        feed_dict = GetFeedDict(df, user_ids)
        cost = session.run(mymodel.loss, feed_dict)
        total_cost += cost
    print total_cost


if __name__ == '__main__':
    if args.mode == 'train':
        Train()
    else:
        Eval()
