# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: disable=protected-access
from pyspark.ml import Estimator
from sparkdl import TextEstimator

from sparkdl.param import keyword_only
from sparkdl.param.shared_params import KafkaParam, FitParam, RunningMode


class TextCNN(Estimator, KafkaParam, FitParam, RunningMode):
    @keyword_only
    def __init__(self, kafkaParam=None, fitParam=None,
                 runningMode="Normal"):
        super(TextCNN, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, kafkaParam=None, fitParam=None,
                  runningMode="Normal"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def fit(self, dataset, params=None):
        estimator = TextEstimator(kafkaParam=self.getKafkaParam(),
                                  runningMode=self.getRunningMode(),
                                  fitParam=self.getFitParam(),
                                  mapFnParam=text_cnn_map_fun)
        return estimator.fit(dataset)


def text_cnn_map_fun(args={}, ctx=None, _read_data=None):
    import tensorflow as tf
    import sys
    import os
    import time

    from sklearn.utils import Bunch
    FLAGS = Bunch(**args["params"]["fitParam"])
    embedded_vec = FLAGS.word_embedding_bs

    num_classes = FLAGS.num_classes
    batch_size = FLAGS.batch_size
    sequence_length = FLAGS.sequence_length

    num_checkpoints = FLAGS.num_checkpoints
    model_name = FLAGS.model_name

    epochs = FLAGS.epochs
    checkpoint_every = FLAGS.checkpoint_every

    per_process_gpu_memory_fraction = FLAGS.per_process_gpu_memory_fraction

    tensor_board_dir = FLAGS.tensor_board_dir
    model_dir = FLAGS.model_dir
    graph_path = FLAGS.graph_path

    dev_collection = FLAGS.dev_collection

    INITIAL_LEARNING_RATE = 0.001
    INITIAL_KEEP_PROB = 0.9

    timestamp = str(int(time.time()))

    out_dir = os.path.abspath(os.path.join(model_dir, timestamp))
    # Checkpoint directory
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    def conv_poo_layer(input, size_in, size_out, filter_width, filter_height, include_pool=False, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([filter_height, filter_width, size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out], name="B"))
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="VALID")

            act = tf.nn.relu(conv + b)

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

            if include_pool:
                return tf.nn.max_pool(act, ksize=[1, filter_height, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            else:
                return act

    def fc_layer(input, size_in, size_out, active="relu", name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W_" + name)
            b = tf.Variable(tf.constant(0.1, shape=[size_out], name="B_" + name))
            if active == "sigmoid":
                act = tf.nn.sigmoid(tf.matmul(input, w) + b)
            elif active is None:
                act = tf.matmul(input, w) + b
            else:
                act = tf.nn.relu(tf.matmul(input, w) + b)
            tf.summary.histogram("W_" + name + "_weights", w)
            tf.summary.histogram("B_" + name + "_biases", b)
            tf.summary.histogram(name + "_activations", act)
            return act

    tf.reset_default_graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    sess = tf.Session(config=config)

    input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

    global_step = tf.Variable(0, name='global_step', trainable=False)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    embeddings = tf.Variable(embedded_vec, dtype=tf.float32)

    def lookup_embedding(word_sequence):
        return tf.nn.embedding_lookup(embeddings, word_sequence)

    embedded_input_x = tf.expand_dims(lookup_embedding(input_x), -1)
    buffer = []

    for vw in [5, 10, 20]:
        conv_layout_num = 0
        pool_layout_num = 0
        conv1 = conv_poo_layer(embedded_input_x, 1, 16, filter_width=EMBEDDING_DIM, filter_height=vw,
                               name="conv1_" + str(vw))
        conv_layout_num += 1
        pool_layout_num += 0

        conv_out = conv_poo_layer(conv1, 16, 32, filter_width=1, filter_height=vw,
                                  name="conv2_" + str(vw))

        conv_layout_num += 1
        pool_layout_num += 0
        flattened = tf.reshape(conv_out, [-1, (
            sequence_length + conv_layout_num + pool_layout_num - conv_layout_num * vw - pool_layout_num * vw) * 32])
        buffer.append(flattened)

    final_flattened = tf.concat(buffer, 1)

    fc1 = fc_layer(final_flattened, int(final_flattened.shape[1]), 1024, "relu", "fc1")
    fc2 = fc_layer(fc1, 1024, 128, "relu", "fc2")
    dropout_fc1 = tf.nn.dropout(fc2, INITIAL_KEEP_PROB)
    _logits = fc_layer(dropout_fc1, 128, num_classes, None, "fc3")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=_logits, labels=input_y), name="xent"
        )
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                                   1200, 0.8, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent, global_step=global_step)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(_logits, 1), tf.argmax(input_y, 1))
        accurate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        tf.summary.scalar("accuracy", accurate)

    summ = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
    tf.train.write_graph(sess.graph_def, graph_path, model_name, as_text=True)

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(tensor_board_dir)
    writer.add_graph(sess.graph)

    writer0 = tf.summary.FileWriter(tensor_board_dir + "/0")
    writer0.add_graph(sess.graph)

    X_TEST = [item["features"] for item in dev_collection]
    Y_TEST = [item["label"] for item in dev_collection]
    for ep in range(epochs):
        print("epoch: %d" % ep)
        for items in _read_data(max_records=batch_size):

            X = [item["features"] for item in items]
            Y = [item["label"].toArray() for item in items]
            _, gs = sess.run([train_step, global_step],
                             feed_dict={input_x: X, input_y: Y, keep_prob: INITIAL_KEEP_PROB})

            [train_accuracy, s, loss] = sess.run([accurate, summ, xent],
                                                 feed_dict={input_x: X, input_y: Y, keep_prob: 1.})

            [test_accuracy, test_s, test_loss] = sess.run([accurate, summ, xent],
                                                          feed_dict={input_x: X_TEST, input_y: Y_TEST, keep_prob: 1.})

            print('train_accuracy %g, test_accuracy %g, loss: %g, global step: %d' % (
                train_accuracy, test_accuracy, loss, gs))

            current_step = tf.train.global_step(sess, global_step)
            if model_dir is not None and current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

            writer.add_summary(s, gs)
            writer0.add_summary(test_s, gs)
            sys.stdout.flush()
    sess.close()
