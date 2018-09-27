# -*- encoding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np


class TextCNN:
    def __init__(self, seq_len, embedding, filter_sizes, num_filters, num_classes, l2_reg=0.0):
        self.seq_len = seq_len
        self.embedding = embedding
        self.embedding_size = embedding.shape[1]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.l2_reg = l2_reg

    def create_placeholder(self):
        self.x = tf.placeholder(tf.int32, shape=[None, self.seq_len], name='x')  # 句子
        self.y = tf.placeholder(tf.int32, shape=[None], name='y')  # 相似度
        self.y_true = tf.one_hot(self.y, depth=4)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss = tf.constant(0.0)
        print(self.x, self.y, self.keep_prob)

    def create_variable(self):
        with tf.name_scope('embedding'):
            self.nonstatic_embedding = tf.Variable(self.embedding, name='nonstatic_embedding')
            self.embedded_chars = tf.nn.embedding_lookup(self.nonstatic_embedding, self.x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)
    
    def create_model(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name='W')
                b = tf.Variable(tf.constant(0.01, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                # h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name='sigmoid')
                # h = tf.nn.softmax(tf.nn.bias_add(conv, b), name='softmax')
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name='tanh')

                pool = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_len-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )
                pooled_outputs.append(pool)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print(self.h_pool)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #"""
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
        print(self.h_drop)
        """

        with tf.name_scope('active'):
            W = tf.Variable(tf.random_normal([num_filters_total, self.hidden_units], stddev=0.01), name='W')
            b = tf.Variable(tf.random_normal([self.hidden_units], stddev=0.1), name='b')
            hidden = tf.nn.tanh(tf.nn.xw_plus_b(self.h_drop, W, b), name='hidden2')
            self.h_drop2 = tf.nn.dropout(hidden, self.keep_prob)
        """

        with tf.name_scope('output'):
            W = tf.get_variable(
                name='W',
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.01, shape=[self.num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

    def create_loss(self, lr=0.001):
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true)
            self.loss = tf.reduce_mean(losses) + self.l2_reg * self.l2_loss
            optimizer = tf.train.GradientDescentOptimizer(lr)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            # '''
            for i, (g, v) in enumerate(grads_and_vars):
                if g is not None:
                    grads_and_vars[i] = (tf.clip_by_norm(g, 5), v)
            # '''
            self.train_op = optimizer.apply_gradients(grads_and_vars)


