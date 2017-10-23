# -*- coding: UTF-8 -*-
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as f
from scipy.sparse import csr_matrix
from sklearn import svm
import numpy as np

from sparkdl.estimators.text_estimator import TextEstimator
from sparkdl.transformers.tf_text import SKLearnTextTransformer


def map_fun(args={}, ctx=None, _read_data=None):
    if ctx is not None:
        raise NotImplementedError("sk-learn do not support distributed training")
    fitParams = args['params']['fitParam']

    # SVM in SK-Learn do not support increment learning yet, so we should collect all data then feeds it.
    def toscipy(v):
        if isinstance(v, DenseVector):
            return csr_matrix((v.values, np.array(range(v.size)), np.array([0, v.size])),
                              shape=(1, v.size))
        elif isinstance(v, SparseVector):
            return csr_matrix((v.values, v.indices, np.array([0, len(v.indices)])),
                              shape=(1, v.size))
        else:
            raise TypeError("Converter.toPandas found unknown Vector type: %s" % type(v))

    X = []
    y = []
    for item in _read_data:
        X.append(toscipy(item["features"]))
        y.append(int(item["labels"]))
    svr = svm.SVC()
    model = svr.fit(X, y)
    print(model)


session = SparkSession.builder.master("local[*]").appName("sk-learn-example").getOrCreate()

documentDF = session.createDataFrame([
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

transformer = SKLearnTextTransformer(
    inputCol="text", outputCol="features")
indexer = StringIndexer(inputCol="preds", outputCol="labels")

pipline = Pipeline(stages=[indexer, transformer])
ds = pipline.fit(documentDF).transform(documentDF)

estimator = TextEstimator(inputCol="features", outputCol="features", labelCol="labels",
                          kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                      "group_id": "sdl_1"},
                          runningMode="Normal",
                          fitParam=[{"epochs": 5, "batch_size": 64}
                                    ],
                          mapFnParam=map_fun)
estimator.fit(ds).collect()
