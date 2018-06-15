import collections
import numpy as np


def GetFeatureVector(tweet_list, vocab, tokenize=False):
    """Get a simple feature vector.

    Args:
      tweet_list: dataframe containing tweets for a user
      vocab: instance of the Vocab class
      tokenize: flag indicating if the tweets are stored as strings 
                or lists of words

    Returns:
      A numpy array of word ids and a numpy array of their corresponding
      counts.
    """
    text = tweet_list['text']
    if tokenize:
      text = text.apply(lambda x: x.split())

    word_counts = collections.Counter()
    for tweet in text:
      word_counts.update([vocab[w] for w in tweet if w in vocab])

    return np.array(word_counts.keys()), np.array(word_counts.values())


def GetEmbed(mymodel, session, feat_vec, counts):
  """Helper function to get the embedding from a feature vector."""
  feed_dict = {
      mymodel.word_ids_pos: feat_vec,
      mymodel.word_count_pos: counts
      }
  embed = session.run(mymodel.embed, feed_dict)
  return embed
