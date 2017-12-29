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
import re
import numpy as np

import jieba
import jieba.posseg
import jieba.analyse

from pyspark.ml import Estimator, Transformer, Pipeline
from pyspark.ml.feature import Word2Vec, Param, Params, TypeConverters, StringIndexer, OneHotEncoder, HashingTF, IDF
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaModel
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.mllib.regression import LabeledPoint

from sparkdl.nlp.text_analysis import TextAnalysis
from sparkdl.param.shared_params import HasEmbeddingSize, HasSequenceLength, HasInputCols, ColumnSuffix, HasOutputCols
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol)
import sparkdl.utils.jvmapi as JVMAPI


class CategoricalBinaryTransformer(Transformer, Estimator, HasInputCols, HasOutputCols, HasEmbeddingSize):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, embeddingSize=12):
        super(CategoricalBinaryTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._setDefault(embeddingSize=12)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, embeddingSize=12):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # def getColumnLabels(self):
    #     return self.column_labels

    def _transform(self, dataset):

        indexer_stages = [
            StringIndexer(**dict(inputCol=item, outputCol=item + "_StringIndex_CategoricalBinaryTransformer")) for
            item in
            self.getInputCols()]

        indexPipeLine = Pipeline(stages=indexer_stages)
        estimator = indexPipeLine.fit(dataset)

        # labels = [model.labels for model in estimator.stages]
        # self.column_labels = {}
        # for column, labels_array in zip(self.getInputCols(), labels):
        #     self.column_labels[column] = labels_array

        dataset_with_index = estimator.transform(dataset)

        def convert_int_to_binary_string(num, length=self.getEmbeddingSize()):
            wow_num = 0
            try:
                wow_num = int(num)
                if wow_num < 0:
                    wow_num = -wow_num
            except:
                wow_num = 0
            return [int(i) for i in ('{0:0' + str(length) + 'b}').format(wow_num)]

        convert_int_to_binary_string_udf = udf(convert_int_to_binary_string, ArrayType(IntegerType()))

        select_expr = [(convert_int_to_binary_string_udf(input + "_StringIndex_CategoricalBinaryTransformer"),
                        self.getOutputCols()[self.getInputCols().index(input)])
                       for
                       input in
                       self.getInputCols()]
        final_ds = dataset_with_index
        for s_expr in select_expr:
            final_ds = final_ds.withColumn(s_expr[1], s_expr[0])
        return final_ds

    def _fit(self, dataset):
        return dataset


class CombineBinaryColumnTransformer(Transformer, Estimator, HasInputCols, HasOutputCol):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(CombineBinaryColumnTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):

        def convert_multi_columns(columns):
            temp = []
            for item in columns:
                for item2 in item:
                    temp.append(item2)
            return temp

        convert_multi_columns_udf = udf(convert_multi_columns, ArrayType(IntegerType()))
        return dataset.withColumn(self.getOutputCol(), convert_multi_columns_udf(array(self.getInputCols())))

    def _fit(self, dataset):
        return dataset


class ElementwiseAdd(Transformer, Estimator, HasInputCols, HasOutputCol):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(ElementwiseAdd, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        def avg_vectors(word_seq):
            result = np.zeros(len(word_seq[0]))
            for item in word_seq:
                result = result + np.array(item)
            return (result / len(word_seq)).tolist()

        avg_vectors_udf = udf(avg_vectors, ArrayType(DoubleType()))
        return dataset.withColumn(self.getOutputCol(), avg_vectors_udf(array(self.getInputCols())))

    def _fit(self, dataset):
        return dataset


class ElementwiseMulti(Transformer, Estimator, HasInputCols, HasOutputCol):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(ElementwiseAdd, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        def multi_vectors(word_seq):
            result = np.ones(len(word_seq[0]))
            for item in word_seq:
                result = result * np.array(item)
            return result.tolist()

        multi_vectors_udf = udf(multi_vectors, ArrayType(DoubleType()))
        return dataset.withColumn(self.getOutputCol(), multi_vectors_udf(array(self.getInputCols())))

    def _fit(self, dataset):
        return dataset


class CategoricalOneHotTransformer(Transformer, Estimator, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(CategoricalOneHotTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        indexer_stages = [
            StringIndexer(**dict(inputCol=item, outputCol=item + "_StringIndexer")) for
            item in
            self.getInputCols()]
        onehot_stages = [OneHotEncoder(
            **dict(inputCol=item + "_StringIndexer",
                   outputCol=self.getOutputCols()[self.getInputCols().index(item)]))
                         for
                         item in self.getInputCols()]
        index_pipeline = Pipeline(stages=indexer_stages)
        onehot_pipeline = Pipeline(stages=onehot_stages)
        dataset_with_index = index_pipeline.fit(dataset).transform(dataset)
        dataset_with_index_and_onehot = onehot_pipeline.fit(dataset_with_index).transform(dataset_with_index)
        return dataset_with_index_and_onehot

    def _fit(self, dataset):
        pass


class TextAnalysisTransformer(Transformer, Estimator, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, stopwords=None, textAnalysisParams=None):
        super(TextAnalysisTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._setDefault(textAnalysisParams={})
        self._setDefault(stopwords=[])
        self.setParams(**kwargs)

    textAnalysisParams = Param(Params._dummy(), "textAnalysisParams", "text analysis params",
                               typeConverter=TypeConverters.identity)

    def setTextAnalysisParams(self, value):
        return self._set(textAnalysisParams=value)

    def getTextAnalysisParams(self):
        return self.getOrDefault(self.textAnalysisParams)

    stopwords = Param(Params._dummy(), "stopwords", "stopwords",
                      typeConverter=TypeConverters.toList)

    def setStopwords(self, value):
        return self._set(stopwords=value)

    def getStopwords(self):
        return self.getOrDefault(self.stopwords)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, stopwords=None, textAnalysisParams=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):

        sc = JVMAPI._curr_sc()
        archiveAutoExtract = sc._conf.get("spark.master").lower().startswith("yarn")
        zipfiles = []
        if not archiveAutoExtract and "dicZipName" in self.getTextAnalysisParams():
            dicZipName = self.getTextAnalysisParams()["dicZipName"]
            if "spark.files" in sc._conf:
                zipfiles = [f.split("/")[-1] for f in sc._conf.get("spark.files").split(",") if
                            f.endswith("{}.zip".format(dicZipName))]

        def analysisParam(name, default_value):
            return self.getTextAnalysisParams()[name] if name in self.getTextAnalysisParams() else default_value

        dicDir = analysisParam("dicDir", "")
        rank = analysisParam("extract_tags", {})
        tmp_dir = analysisParam("tmp_dir", None)
        char_mode = analysisParam("char_mode", False)
        stopwords_br = sc.broadcast(self.getStopwords())

        def lcut(s):
            TextAnalysis.load_dic(dicDir, archiveAutoExtract, zipfiles, tmp_dir)
            words = []

            def _cut():
                if len(rank) > 0:
                    if "type" not in rank or rank["type"] == "textrank":
                        words = jieba.analyse.textrank(s, topK=rank["topK"], withWeight=False)
                    if "type" in rank and rank["type"] == "tfidf":
                        words = jieba.analyse.tfidf(s)
                else:
                    words = jieba.lcut(s)
                return words

            def _chars():
                return [word for word in unicode(s, "utf-8")]

            if char_mode:
                words = _cut()
            else:
                words = _chars()

            return [word for word in words if word.lower() not in stopwords_br.value]

        lcut_udf = udf(lcut, ArrayType(StringType()))

        select_expr = [(lcut_udf(input), self.getOutputCols()[self.getInputCols().index(input)]) for input in
                       self.getInputCols()]
        final_ds = dataset
        for s_expr in select_expr:
            final_ds = final_ds.withColumn(s_expr[1], s_expr[0])
        return final_ds

    def _fit(self, dataset):
        pass


class TextTFDFTransformer(Transformer, HasInputCols, HasOutputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, numFeatures=None):
        super(TextTFDFTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._setDefault(numFeatures=10000)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, numFeatures=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    numFeatures = Param(Params._dummy(), "numFeatures", "numFeatures",
                        typeConverter=TypeConverters.toInt)

    def setNumFeatures(self, value):
        return self._set(numFeatures=value)

    def getNumFeatures(self):
        return self.getOrDefault(self.numFeatures)

    def idfModel(self):
        return self.idfModel

    def _transform(self, dataset):
        column_initial = self.getInputCols()[0]
        ds = dataset.select(col(column_initial).alias("tfidf_TextTFDFTransformer_inputCol"))
        for column in [item for item in self.getInputCols() if item not in [column_initial]]:
            ds = ds.union(dataset.select(col(column).alias("tfidf_TextTFDFTransformer_inputCol")))

        hashingTF = HashingTF(inputCol="tfidf_TextTFDFTransformer_inputCol",
                              outputCol="tfidf_TextTFDFTransformer_outputCol",
                              numFeatures=self.getNumFeatures())

        featurizedData = hashingTF.transform(ds)
        # alternatively, CountVectorizer can also be used to get term frequency vectors
        idf = IDF(inputCol="tfidf_TextTFDFTransformer_outputCol", outputCol="tfidf_TextTFDFTransformer_idf_outputCol")
        self.idfModel = idf.fit(featurizedData)

        hashingTFs = [HashingTF(inputCol=input,
                                outputCol=input + "_TextTFDFTransformer",
                                numFeatures=self.getNumFeatures()) for input in self.getInputCols()]

        pipline = Pipeline(stages=hashingTFs)
        featurizedData2 = pipline.fit(dataset).transform(dataset)
        for input in self.getInputCols():
            self.idfModel._call_java("setInputCol", input + "_TextTFDFTransformer")
            self.idfModel._call_java("setOutputCol", self.getOutputCols()[
                self.getInputCols().index(input)])
            featurizedData2 = self.idfModel.transform(featurizedData2)
        return featurizedData2


class TextEmbeddingSequenceTransformer(Transformer, Estimator, HasInputCols, HasOutputCols, HasEmbeddingSize,
                                       HasSequenceLength):
    def _fit(self, dataset):
        return dataset

    VOCAB_SIZE = 'vocab_size'
    EMBEDDING_SIZE = 'embedding_size'

    textAnalysisParams = Param(Params._dummy(), "textAnalysisParams", "text analysis params",
                               typeConverter=TypeConverters.identity)

    def setTextAnalysisParams(self, value):
        return self._set(textAnalysisParams=value)

    def getTextAnalysisParams(self):
        return self.getOrDefault(self.textAnalysisParams)

    wordEmbeddingSavePath = Param(Params._dummy(), "wordEmbeddingSavePath", "",
                                  typeConverter=TypeConverters.toString)

    def setWordEmbeddingSavePath(self, value):
        return self._set(wordEmbeddingSavePath=value)

    def getWordEmbeddingSavePath(self):
        return self.getOrDefault(self.wordEmbeddingSavePath)

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, embeddingSize=100, sequenceLength=64,
                 wordEmbeddingSavePath=None):
        super(TextEmbeddingSequenceTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._setDefault(sequenceLength=64)
        self._setDefault(embeddingSize=100)
        self._setDefault(wordEmbeddingSavePath=None)
        # self._setDefault()
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, embeddingSize=100, sequenceLength=64,
                  wordEmbeddingSavePath=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getWordEmbedding(self):
        return self.word_embedding_with_index

    def getW2vModel(self):
        return self.w2v_model

    def _transform(self, dataset):
        sc = JVMAPI._curr_sc()
        column_initial = self.getInputCols()[0]
        ds = dataset.select(col(column_initial).alias("word2vec_TextEmbeddingSequenceTransformer"))
        for column in [item for item in self.getInputCols() if item not in [column_initial]]:
            ds = ds.union(dataset.select(col(column).alias("word2vec_TextEmbeddingSequenceTransformer")))

        word2vec = Word2Vec(vectorSize=self.getEmbeddingSize(), minCount=1,
                            inputCol="word2vec_TextEmbeddingSequenceTransformer",
                            outputCol="person_behavior_vector")
        self.w2v_model = word2vec.fit(ds)
        self.word_embedding = self.w2v_model.getVectors().rdd.map(
            lambda p: dict(word=p.word, vector=p.vector.values.tolist())).collect()

        self.word_embedding_with_index = [dict(word_index=(idx + 1), word=val["word"], vector=val["vector"]) for
                                          (idx, val)
                                          in
                                          enumerate(self.word_embedding)]
        self.word_embedding_with_index.insert(0, dict(word_index=0, word="UNK",
                                                      vector=np.zeros(self.getEmbeddingSize()).tolist()))

        if self.getWordEmbeddingSavePath() is not None:
            SparkSession.builder.getOrCreate().createDataFrame(self.word_embedding_with_index).write.mode(
                "overwrite").format(
                "parquet").save(self.getWordEmbeddingSavePath())

        # word_embedding_bs = sc.broadcast(word_embedding)
        word2index_bs = sc.broadcast(
            dict([(item["word"], item["word_index"]) for item in self.word_embedding_with_index]))

        # index2word_bs = sc.broadcast(dict([(item["word_index"], item["word"]) for item in word_embedding_with_index]))
        sequence_len = self.getSequenceLength()

        def sentence_sequence(s):
            new_s = [word2index_bs.value[word] for word in s if word in word2index_bs.value]

            def _pad_sequences(sequences, maxlen=sequence_len):
                new_sequences = []

                if len(sequences) <= maxlen:
                    for i in range(maxlen - len(sequences)):
                        new_sequences.append(0)
                    return sequences + new_sequences
                else:
                    return sequences[0:maxlen]

            return _pad_sequences(new_s)

        sentence_sequence_udf = udf(sentence_sequence, ArrayType(IntegerType()))
        select_expr = [(sentence_sequence_udf(input), self.getOutputCols()[self.getInputCols().index(input)]) for input
                       in self.getInputCols()]
        final_ds = dataset
        for s_expr in select_expr:
            final_ds = final_ds.withColumn(s_expr[1], s_expr[0])
        return final_ds


class TFTextTransformer(Transformer, Estimator, HasInputCol, HasOutputCol, HasEmbeddingSize, HasSequenceLength):
    """
    Convert sentence/document to a 2-D Array eg. [[word embedding],[....]]  in DataFrame which can be processed
    directly by tensorflow or keras who's backend is tensorflow.

    Processing Steps:

    * Using Word2Vec compute Map(word -> vector) from input column, then broadcast the map.
    * Process input column (which is text),split it with white space, replace word with vector, padding the result to
      the same size.
    * Create a new dataframe with columns like new 2-D array , vocab_size, embedding_size
    * return then new dataframe
    """

    def _fit(self, dataset):
        pass

    VOCAB_SIZE = 'vocab_size'
    EMBEDDING_SIZE = 'embedding_size'

    textAnalysisParams = Param(Params._dummy(), "textAnalysisParams", "text analysis params",
                               typeConverter=TypeConverters.identity)

    def setTextAnalysisParams(self, value):
        return self._set(textAnalysisParams=value)

    def getTextAnalysisParams(self):
        return self.getOrDefault(self.textAnalysisParams)

    shape = Param(Params._dummy(), "shape", "result shape",
                  typeConverter=TypeConverters.identity)

    def setShape(self, value):
        return self._set(shape=value)

    def getShape(self):
        return self.getOrDefault(self.shape)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, textAnalysisParams={}, shape=(64, 100), embeddingSize=100,
                 sequenceLength=64):
        super(TFTextTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._setDefault(textAnalysisParams={})
        self._setDefault(embeddingSize=100)
        self._setDefault(sequenceLength=64)
        self._setDefault(shape=(self.getSequenceLength(), self.getEmbeddingSize()))
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, textAnalysisParams={}, shape=(64, 100), embeddingSize=100,
                  sequenceLength=64):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        sc = JVMAPI._curr_sc()

        word2vec = Word2Vec(vectorSize=self.getEmbeddingSize(), minCount=1, inputCol=self.getInputCol(),
                            outputCol="word_embedding")

        archiveAutoExtract = sc._conf.get("spark.master").lower().startswith("yarn")
        zipfiles = []
        if not archiveAutoExtract and "dicZipName" in self.getTextAnalysisParams():
            dicZipName = self.getTextAnalysisParams()["dicZipName"]
            if "spark.files" in sc._conf:
                zipfiles = [f.split("/")[-1] for f in sc._conf.get("spark.files").split(",") if
                            f.endswith("{}.zip".format(dicZipName))]

        dicDir = self.getTextAnalysisParams()["dicDir"] if "dicDir" in self.getTextAnalysisParams() else ""

        def lcut(s):
            TextAnalysis.load_dic(dicDir, archiveAutoExtract, zipfiles)
            return jieba.lcut(s)

        lcut_udf = udf(lcut, ArrayType(StringType()))
        vectorsDf = word2vec.fit(
            dataset.select(lcut_udf(self.getInputCol()).alias(self.getInputCol()))).getVectors()

        """
          It's strange here that after calling getVectors the df._sc._jsc will lose and this is
          only happens when you run it with ./python/run-tests.sh script.
          We add this code to make it pass the test. However it seems this will hit
          "org.apache.spark.SparkException: EOF reached before Python server acknowledged" error.
        """
        if vectorsDf._sc._jsc is None:
            vectorsDf._sc._jsc = sc._jsc

        word_embedding = dict(vectorsDf.rdd.map(
            lambda p: (p.word, p.vector.values.tolist())).collect())

        word_embedding["unk"] = np.zeros(self.getEmbeddingSize()).tolist()
        local_word_embedding = sc.broadcast(word_embedding)

        not_array_2d = len(self.getShape()) != 2 or self.getShape()[0] != self.getSequenceLength()

        def convert_word_to_index(s):
            def _pad_sequences(sequences, maxlen=None):
                new_sequences = []

                if len(sequences) <= maxlen:
                    for i in range(maxlen - len(sequences)):
                        new_sequences.append(np.zeros(self.getEmbeddingSize()).tolist())
                    return sequences + new_sequences
                else:
                    return sequences[0:maxlen]

            new_q = [local_word_embedding.value[word] for word in re.split(r"\s+", s) if
                     word in local_word_embedding.value.keys()]
            result = _pad_sequences(new_q, maxlen=self.getSequenceLength())
            if not_array_2d:
                result = np.array(result).reshape(self.getShape()).tolist()
            return result

        cwti_udf = udf(convert_word_to_index, ArrayType(ArrayType(FloatType())))

        if not_array_2d:
            cwti_udf = udf(convert_word_to_index, ArrayType(FloatType()))

        doc_martic = (dataset.withColumn(self.getOutputCol(), cwti_udf(self.getInputCol()).alias(self.getOutputCol()))
                      .withColumn(self.VOCAB_SIZE, lit(len(word_embedding)))
                      .withColumn(self.EMBEDDING_SIZE, lit(self.getEmbeddingSize()))
                      )

        return doc_martic
