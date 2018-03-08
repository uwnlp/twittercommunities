""" Module to store Model"""
import tensorflow as tf

class Model(object):
    """
    Top level Model class
    Defines embedding placeholders and optimizer
    """

    def GetEmbedding(self, word_ids, counts):
        """Helper function to create user embedding from bag-of-words representation."""

        if self.weighting == 'log':
            weighted_counts = tf.log(tf.to_float(counts + 1))
        elif self.weighting == 'tfidf':
            weighted_counts = tf.multiply(tf.log(tf.to_float(counts + 1)),
                                          -tf.log(self.idf_weights))
        else:
            weighted_counts = tf.to_float(counts)

        wembeds = tf.nn.embedding_lookup(self.word_embeddings, word_ids)

        wembeds = tf.multiply(tf.transpose(wembeds), weighted_counts)

        if self.projection_mat:  # use this when embeddings are frozen
            wembeds = tf.matmul(self.projection_mat, wembeds)

        final_embed = tf.transpose(self.pooler(wembeds, 1))
        normalized_embed = tf.nn.l2_normalize(final_embed, 0)

        # For debugging get the top 5 words (only works with max pooling)
        idxs = tf.argmax(wembeds, 0)
        _, top_positions = tf.nn.top_k(final_embed, 5)
        top_ids = tf.gather(word_ids, tf.gather(idxs, top_positions))

        return normalized_embed, top_ids


    def __init__(self, params):
        self.weighting = params.get('weighting', 'log')
        weights = []
        if self.weighting == 'tfidf':
            with open('tfidf.txt', 'r') as f:
                for line in f:
                    fields = line.split('\t')
                if len(fields) == 2:
                    weights.append(float(fields[-1]))
        self.idf_weights = tf.constant(weights, name='idf')

        """The anchor should be from the same person as the pos placeholder and
        the neg should be a different person."""
        self.word_ids_pos = tf.placeholder(tf.int64, [None], name='word_ids_pos')
        self.word_ids_neg = tf.placeholder(tf.int64, [None], name='word_ids_neg')
        self.word_ids_anchor = tf.placeholder(tf.int64, [None], name='word_ids_anchor')

        self.word_count_pos = tf.placeholder(tf.int64, [None,], name='word_count_pos')
        self.word_count_neg = tf.placeholder(tf.int64, [None,], name='word_count_neg')
        self.word_count_anchor = tf.placeholder(tf.int64, [None,], name='word_count_anchor')

        pool_funcs = {'max': tf.reduce_max, 'mean': tf.reduce_mean}
        self.pooler = pool_funcs[params.get('pooling', 'mean')]  # default is mean

        self.embedding_dim = 128
        self.word_embeddings = tf.get_variable('word_embeddings',
                                               [params['vocab_size'],
                                                self.embedding_dim],
                                               trainable=not params['freeze_embeddings'])

        self.projection_mat = None
        if params['freeze_embeddings']:
            self.projection_mat = tf.get_variable(
                'projection_mat', [self.embedding_dim, self.embedding_dim])

        # Multiply each embedding element wise by its appropriately weighted count place holder
        pos_embed, self.top5 = self.GetEmbedding(self.word_ids_pos, self.word_count_pos)
        self.embed = pos_embed
        neg_embed, _ = self.GetEmbedding(self.word_ids_neg, self.word_count_neg)
        anchor_embed, _ = self.GetEmbedding(self.word_ids_anchor, self.word_count_anchor)

        if params['cos_dist']:
            self.d_pos = 1.0 - tf.reduce_sum(tf.multiply(pos_embed, anchor_embed))
            self.d_neg = 1.0 - tf.reduce_sum(tf.multiply(neg_embed, anchor_embed))
        else:
            self.d_pos = tf.sqrt(tf.nn.l2_loss(pos_embed - anchor_embed))
            self.d_neg = tf.sqrt(tf.nn.l2_loss(neg_embed - anchor_embed))

        # Define the margin loss
        margin = params['margin']
        self.loss = tf.nn.relu(margin + self.d_pos - self.d_neg)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)
