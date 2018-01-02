# -*- coding: UTF-8 -*-
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

import shutil
import threading
import sys

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Tokenizer
from pyspark.sql.types import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sparkdl.transformers.easy_feature import EasyFeature
from sparkdl.transformers.tf_text import CategoricalBinaryTransformer, CategoricalOneHotTransformer, \
    TextAnalysisTransformer, TextEmbeddingSequenceTransformer, CombineBinaryColumnTransformer, TextTFDFTransformer
from sparkdl.estimators.text_estimator import TextEstimator, KafkaMockServer
from sparkdl.transformers.tf_text import TFTextTransformer
from ..tests import SparkDLTestCase

if sys.version_info[:2] <= (2, 7):
    import cPickle as pickle
else:
    import _pickle as pickle


def map_fun(args={}, ctx=None, _read_data=None):
    import tensorflow as tf
    params = args['params']['fitParam']
    EMBEDDING_SIZE = params["embedding_size"]
    SEQUENCE_LENGTH = 64

    def feed_dict(batch):
        # Convert from dict of named arrays to two numpy arrays of the proper type
        features = []
        for i in batch:
            features.append(i['sentence_matrix'])

        # print("{} {}".format(feature, features))
        return features

    encoder_variables_dict = {
        "encoder_w1": tf.Variable(
            tf.random_normal([SEQUENCE_LENGTH * EMBEDDING_SIZE, 256]), name="encoder_w1"),
        "encoder_b1": tf.Variable(tf.random_normal([256]), name="encoder_b1"),
        "encoder_w2": tf.Variable(tf.random_normal([256, 128]), name="encoder_w2"),
        "encoder_b2": tf.Variable(tf.random_normal([128]), name="encoder_b2")
    }

    def encoder(x, name="encoder"):
        with tf.name_scope(name):
            encoder_w1 = encoder_variables_dict["encoder_w1"]
            encoder_b1 = encoder_variables_dict["encoder_b1"]

            layer_1 = tf.nn.sigmoid(tf.matmul(x, encoder_w1) + encoder_b1)

            encoder_w2 = encoder_variables_dict["encoder_w2"]
            encoder_b2 = encoder_variables_dict["encoder_b2"]

            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, encoder_w2) + encoder_b2)
            return layer_2

    def decoder(x, name="decoder"):
        with tf.name_scope(name):
            decoder_w1 = tf.Variable(tf.random_normal([128, 256]))
            decoder_b1 = tf.Variable(tf.random_normal([256]))

            layer_1 = tf.nn.sigmoid(tf.matmul(x, decoder_w1) + decoder_b1)

            decoder_w2 = tf.Variable(
                tf.random_normal([256, SEQUENCE_LENGTH * EMBEDDING_SIZE]))
            decoder_b2 = tf.Variable(
                tf.random_normal([SEQUENCE_LENGTH * EMBEDDING_SIZE]))

            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, decoder_w2) + decoder_b2)
            return layer_2

    tf.reset_default_graph
    sess = tf.Session()

    input_x = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, EMBEDDING_SIZE], name="input_x")
    flattened = tf.reshape(input_x,
                           [-1, SEQUENCE_LENGTH * EMBEDDING_SIZE])

    encoder_op = encoder(flattened)

    tf.add_to_collection('encoder_op', encoder_op)

    y_pred = decoder(encoder_op)

    y_true = flattened

    with tf.name_scope("xent"):
        consine = tf.div(tf.reduce_sum(tf.multiply(y_pred, y_true), 1),
                         tf.multiply(tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, y_pred), 1)),
                                     tf.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1))))
        xent = tf.reduce_sum(tf.subtract(tf.constant(1.0), consine))
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(xent)
        train_step = tf.train.RMSPropOptimizer(0.01).minimize(xent)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(params["epochs"]):
        print("epoll {}".format(i))
        for data in _read_data(max_records=params["batch_size"]):
            batch_data = feed_dict(data)
            sess.run(train_step, feed_dict={input_x: batch_data})

    sess.close()


class EasyFeatureTest(SparkDLTestCase):
    def test_trainText(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "Hi I heard about Spark", 2.0, 3.0, 1, 2),
            ("I wish Java could use case classes", "I wish Java could use case classes", 3.0, 4.0, 0, 4),
            ("Logistic regression models are neat", "Logistic regression models are neat", 4.0, 5.0, 2, 5)
        ], ["sentence", "sentence2", "f1", "f2", "preds", "i1"])
        tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
        wordsData = tokenizer.transform(documentDF)
        tokenizer = Tokenizer(inputCol="sentence2", outputCol="words2")
        wordsData2 = tokenizer.transform(wordsData)
        # transform text column to sentence_matrix column which contains 2-D array.
        # transformer = TextTFDFTransformer(inputCols=["words", "words2"], outputCols=["text_tfidf", "text_tfidf2"],
        #                                   numFeatures=10)
        # df = transformer.transform(wordsData2)
        # df.show()
        ef = EasyFeature(textFields=["sentence", "sentence2"], numFeatures=10, outputCol="features", wordMode="tfidf",
                         discretizerFields={"f1": 2}, outputColPNorm=2)
        df = ef.transform(documentDF)
        df.select("features").show(truncate=False)

        ef = EasyFeature(textFields=["sentence"], excludeFields=["sentence2", "f1", "f2", "preds", "i1"],
                         outputCol="features")

        df = ef.transform(documentDF)
        df.show(truncate=False)

        onehot = CategoricalOneHotTransformer(inputCols=["preds"], outputCols=["pc"])
        onehot.transform(df).show()


class OnlyForTest(SparkDLTestCase):
    def test(self):
        documentDF = self.session.createDataFrame([
            (u"Hi I heard about Spark，中国", "Hi I heard about Spark", 2.0, 3.0, 1, 2),
            (u"I wish Java could use case classes", "I wish Java could use case classes", 3.0, 4.0, 0, 4),
            ("Logistic regression models are neat", "Logistic regression models are neat", 4.0, 5.0, 2, 5)
        ], ["sentence", "sentence2", "f1", "f2", "preds", "i1"])

        ef = EasyFeature(textFields=["sentence"], excludeFields=["sentence2", "f1", "f2", "preds", "i1"],
                         textAnalysisParams={"char_mode": True},
                         outputCol="features")
        df = ef.transform(documentDF)
        word_embedding = [item["word"] for item in ef.getWordEmbedding()]
        seqs = [item["sentence_text_EasyFeature"] for item in df.collect()]
        words = [word_embedding[item] for item in seqs[0]]
        print(words)


class EasyFeatureBasicFunctionTest(SparkDLTestCase):
    def test_getConvertMapping(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "Hi I heard about Spark", 2.0, 3.0, 1, 2, "yes"),
            ("I wish Java could use case classes", "I wish Java could use case classes", 3.0, 4.0, 0, 4, "cool"),
            ("Logistic regression models are neat", "Logistic regression models are neat", 4.0, 5.0, 2, 5, "wow")
        ], ["sentence", "sentence2", "f1", "f2", "preds", "i1", "label"])

        ef = EasyFeature(excludeFields=["sentence2", "f1", "f2", "preds", "i1"], outputCol="features")
        df = ef.transform(documentDF)
        print ef.getConvertMapping(df, "label")


class TFTextEstimatorTest(SparkDLTestCase):
    def test_trainText(self):
        input_col = "text"
        output_col = "sentence_matrix"

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        # transform text column to sentence_matrix column which contains 2-D array.
        transformer = TFTextTransformer(
            inputCol=input_col, outputCol=output_col, embeddingSize=100, sequenceLength=64)

        df = transformer.transform(documentDF)
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()
        # create a estimator to training where map_fun contains tensorflow's code
        estimator = TextEstimator(kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                              "mock_kafka_file": mock_kafka_file,
                                              "group_id": "sdl_1", "test_mode": True},
                                  runningMode="Normal",
                                  fitParam=[{"epochs": 5, "batch_size": 64, "embedding_size": 100}],
                                  mapFnParam=map_fun)
        estimator.fit(df).collect()
        shutil.rmtree(mock_kafka_file)


class A(SparkDLTestCase):
    def test_a(self):
        df = self.session.range(0, 100)
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()

        def kk(args={}, ctx=None, _read_data=None):
            count = 0
            for data in _read_data(max_records=10):
                print(data)
                count += 1
                if count > 100:
                    break

        # create a estimator to training where map_fun contains tensorflow's code
        estimator = TextEstimator(kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                              "mock_kafka_file": mock_kafka_file,
                                              "group_id": "sdl_1", "test_mode": True},
                                  runningMode="Normal",
                                  fitParam=[{"epochs": 5, "batch_size": 64, "embedding_size": 100}],
                                  mapFnParam=kk)
        estimator.fit(df).collect()
        shutil.rmtree(mock_kafka_file)


class TFTextTransformerSkLearnTest(SparkDLTestCase):
    def test_trainText(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "spark"),
            ("I wish Java could use case classes", "java"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "mlib")
        ], ["text", "preds"])

        features = TFTextTransformer(
            inputCol="text", outputCol="features", shape=(-1,), embeddingSize=100, sequenceLength=64)
        ds = features.transform(documentDF)
        result = ds.take(1)[0]
        self.assertTrue(len(result["features"]) == 100 * 64)


class ExampleTransformerTest(SparkDLTestCase):
    def test_trainText(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "spark"),
            ("I wish Java could use case classes", "java"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "mlib")
        ], ["text", "preds"])
        tat = TextAnalysisTransformer(
            inputCols=["text", "preds"], outputCols=["text1", "preds2"],
            textAnalysisParams={"extract_tags": {"type": "tfidf"}})
        ds = tat.transform(documentDF)
        ds.show()


class CategoricalTransformerTest(SparkDLTestCase):
    def test_trainText(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "spark"),
            ("I wish Java could use case classes", "java"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "mlib")
        ], ["text", "preds"])

        features = CategoricalBinaryTransformer(
            inputCols=["text", "preds"], outputCols=["text1", "preds1"], embeddingSize=12)
        ds = features.transform(documentDF)
        ds.show()
        result = ds.take(1)[0]
        self.assertTrue(len(result["text1"]) == 12)

        cbct = CombineBinaryColumnTransformer(
            inputCols=["text1", "preds1"], outputCol="text2")
        ds = cbct.transform(ds)
        ds.show()
        result = ds.take(1)[0]
        self.assertTrue(len(result["text2"]) == 24)

        features = CategoricalOneHotTransformer(
            inputCols=["text", "preds"], outputCol="features")
        ds = features.transform(documentDF)
        ds.show()
        result = ds.take(1)[0]
        # self.assertTrue(len(result["features"]) == 24)

        tat = TextAnalysisTransformer(
            inputCols=["text", "preds"], outputCols=["text1", "preds2"])
        ds = tat.transform(documentDF)
        ds.show()

        test = TextEmbeddingSequenceTransformer(
            inputCols=["text1", "preds2"], outputCols=["text11", "preds21"])
        ds2 = test.transform(ds)
        ds2.show()


class TFTextEstimatorSkLearnTest(SparkDLTestCase):
    def test_trainText(self):
        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", "spark"),
            ("I wish Java could use case classes", "java"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "mlib"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "spark"),
            ("Logistic regression models are neat", "java"),
            ("Logistic regression models are neat", "mlib")
        ], ["text", "preds"])

        # transform text column to sentence_matrix column which contains 2-D array.
        features = TFTextTransformer(
            inputCol="text", outputCol="features", shape=(-1,), embeddingSize=100, sequenceLength=64)

        indexer = StringIndexer(inputCol="preds", outputCol="labels")

        pipline = Pipeline(stages=[features, indexer])
        ds = pipline.fit(documentDF).transform(documentDF)
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()

        def sk_map_fun(args={}, ctx=None, _read_data=None):
            data = [item for item in _read_data()]
            parameters = {'kernel': ('linear', 'rbf')}
            svr = svm.SVC()
            clf = GridSearchCV(svr, parameters)
            X = [x["features"] for x in data[0]]
            y = [int(x["labels"]) for x in data[0]]
            model = clf.fit(X, y)
            print(model.best_estimator_)
            return ""

        # create a estimator to training where map_fun contains tensorflow's code
        estimator = TextEstimator(kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                              "mock_kafka_file": mock_kafka_file,
                                              "group_id": "sdl_1", "test_mode": False},
                                  runningMode="Normal",
                                  fitParam=[{"epochs": 5, "batch_size": 64}, {"epochs": 5, "batch_size": 1}],
                                  mapFnParam=sk_map_fun)
        estimator.fit(ds).collect()
        shutil.rmtree(mock_kafka_file)


class MockKakfaServerTest(SparkDLTestCase):
    def test_mockKafkaServerProduce(self):
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()
        dataset = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        def _write_data():
            def _write_partition(index, d_iter):
                producer = KafkaMockServer(index, mock_kafka_file)
                try:
                    for d in d_iter:
                        producer.send("", pickle.dumps(d))
                    producer.send("", pickle.dumps("_stop_"))
                    producer.flush()
                finally:
                    producer.close()
                return []

            dataset.rdd.mapPartitionsWithIndex(_write_partition).count()

        _write_data()

        def _consume():
            consumer = KafkaMockServer(0, mock_kafka_file)
            stop_count = 0
            while True:
                messages = consumer.poll(timeout_ms=1000, max_records=64)
                group_msgs = []
                for tp, records in messages.items():
                    for record in records:
                        try:
                            msg_value = pickle.loads(record.value)
                            print(msg_value)
                            if msg_value == "_stop_":
                                stop_count += 1
                            else:
                                group_msgs.append(msg_value)
                        except:
                            pass
                if stop_count >= 8:
                    break
            self.assertEquals(stop_count, 8)

            t = threading.Thread(target=_consume)
            t.start()
            t2 = threading.Thread(target=_consume)
            t2.start()
            import time
            time.sleep(10)
            shutil.rmtree(mock_kafka_file)
