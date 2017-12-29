from pyspark.ml import Estimator, Transformer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, Param, Params, TypeConverters, VectorAssembler, VectorIndexer, Tokenizer, \
    HashingTF, OneHotEncoder, QuantileDiscretizer, Normalizer
from pyspark.ml.linalg import VectorUDT, Vectors, DenseVector
from pyspark.sql.types import *
import pyspark.sql.functions as fn

from sparkdl.param.shared_params import HasEmbeddingSize, HasSequenceLength, HasOutputCol
from sparkdl.param import keyword_only
from sparkdl.transformers.tf_text import CategoricalBinaryTransformer, TextAnalysisTransformer, \
    TextEmbeddingSequenceTransformer, TextTFDFTransformer


class EasyFeature(Transformer, HasEmbeddingSize, HasSequenceLength, HasOutputCol):
    @keyword_only
    def __init__(self, textFields=None, excludeFields=None, stopwords=[], wordMode=None, discretizerFields=None,
                 numFeatures=10000,
                 maxCategories=20,
                 embeddingSize=100,
                 sequenceLength=64,
                 wordEmbeddingSavePath=None, outputCol=None, outputColPNorm=None, textAnalysisParams=None):
        super(EasyFeature, self).__init__()
        kwargs = self._input_kwargs
        self.ignoredColumns = []
        self._setDefault(sequenceLength=64)
        self._setDefault(embeddingSize=100)
        self._setDefault(wordEmbeddingSavePath=None)
        self._setDefault(textFields=[])
        self._setDefault(outputCol=None)
        self._setDefault(maxCategories=20)
        self._setDefault(wordMode="embedding")
        self._setDefault(numFeatures=10000)
        self._setDefault(discretizerFields={})
        self._setDefault(outputColPNorm=None)
        self._setDefault(excludeFields=[])
        self._setDefault(stopwords=[])
        self._setDefault(textAnalysisParams={})
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, textFields=None, excludeFields=None, stopwords=[], wordMode=None, discretizerFields=None,
                  numFeatures=10000, maxCategories=20, embeddingSize=100,
                  sequenceLength=64,
                  wordEmbeddingSavePath=None, outputCol=None, outputColPNorm=None, textAnalysisParams=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    textFields = Param(Params._dummy(), "textFields", "Specify which fields should be segmented",
                       typeConverter=TypeConverters.toList)
    excludeFields = Param(Params._dummy(), "excludeFields", "Specify which fields should be excluded",
                          typeConverter=TypeConverters.toList)
    stopwords = Param(Params._dummy(), "stopwords", "stopwords",
                      typeConverter=TypeConverters.toList)

    outputColPNorm = Param(Params._dummy(), "outputColPNorm",
                           "Specifies the p-norm used for normalization which is performed in outputCol",
                           typeConverter=TypeConverters.toInt)

    wordMode = Param(Params._dummy(), "wordMode",
                     "wordMode: embedding or tfidf. default: embedding. when embedding is selected , " +
                     "the text field will be convert to int sequence and " +
                     "you should also specify embeddingSize and sequenceLength params. " +
                     "if tfid is selected , the text fields will be convert to tf/idf sequence which length is specified by numFeatures",
                     typeConverter=TypeConverters.toString)

    discretizerFields = Param(Params._dummy(), "discretizerFields",
                              "fields which are continuous will be convert to categorical . here is the example:" +
                              "{\"field1\":3,\"field2\":2,} . " +
                              "the keys are field  name and the value means the coresponding field have how many buckets ",
                              typeConverter=TypeConverters.identity)

    textAnalysisParams = Param(Params._dummy(), "textAnalysisParams", "text analysis params",
                               typeConverter=TypeConverters.identity)

    wordEmbeddingSavePath = Param(Params._dummy(), "wordEmbeddingSavePath", "",
                                  typeConverter=TypeConverters.toString)

    numFeatures = Param(Params._dummy(), "numFeatures",
                        "Specify the size of vector  when use tf/idf algorithm in text fields.",
                        typeConverter=TypeConverters.toInt)

    maxCategories = Param(Params._dummy(), "maxCategories",
                          "Threshold for the number of values a categorical feature can take " +
                          "(>= 2). If a feature is found to have > maxCategories values, then " +
                          "it is declared continuous.", typeConverter=TypeConverters.toInt)

    def setStopwords(self, value):
        return self._set(stopwords=value)

    def getStopwords(self):
        return self.getOrDefault(self.stopwords)

    def setExcludeFields(self, value):
        return self._set(excludeFields=value)

    def getExcludeFields(self):
        return self.getOrDefault(self.excludeFields)

    def setOutputColPNorm(self, value):
        return self._set(outputColPNorm=value)

    def getOutputColPNorm(self):
        return self.getOrDefault(self.outputColPNorm)

    def setDiscretizerFields(self, value):
        return self._set(discretizerFields=value)

    def getDiscretizerFields(self):
        return self.getOrDefault(self.discretizerFields)

    def setNumFeatures(self, value):
        return self._set(numFeatures=value)

    def getNumFeatures(self):
        return self.getOrDefault(self.numFeatures)

    def setTextFields(self, value):
        return self._set(textFields=value)

    def getTextFields(self):
        return self.getOrDefault(self.textFields)

    VOCAB_SIZE = 'vocab_size'
    EMBEDDING_SIZE = 'embedding_size'

    def setTextAnalysisParams(self, value):
        return self._set(textAnalysisParams=value)

    def getTextAnalysisParams(self):
        return self.getOrDefault(self.textAnalysisParams)

    def setWordMode(self, value):
        return self._set(wordMode=value)

    def getWordMode(self):
        return self.getOrDefault(self.wordMode)

    def setWordEmbeddingSavePath(self, value):
        return self._set(wordEmbeddingSavePath=value)

    def getWordEmbeddingSavePath(self):
        return self.getOrDefault(self.wordEmbeddingSavePath)

    def getWordEmbedding(self):
        return self.word_embedding_with_index

    def getW2vModel(self):
        return self.w2v_model

    def setMaxCategories(self, value):
        """
        Sets the value of :py:attr:`maxCategories`.
        """
        return self._set(maxCategories=value)

    def getMaxCategories(self):
        """
        Gets the value of maxCategories or its default value.
        """
        return self.getOrDefault(self.maxCategories)

    def getCategoricalFeatures(self):
        return self.categoricalFeatures

    def getDiscretizerFeatures(self):
        return self.discretizerFeatures

    def getIgnoredColumns(self):
        return self.ignoredColumns

    def getConvertMapping(self, df, field):
        """
        :param df:
        :param field: original field
        :return:  mapping from original field to new field

        EasyFeature will do some feature engineer jobs like:
         1. convert String to Int
         2. convert word to word embbedding
         3. more....
         We should get the mapping when we are in predicting step.
        """
        original_field = field
        convert_field = original_field + "_EasyFeature"
        mapping = df.select(original_field, convert_field).groupby(field).agg(
            fn.first(convert_field).alias(convert_field)).collect()
        return dict([(item[field], item[convert_field]) for item in mapping])

    def _transform(self, dataset):
        st = dataset.schema

        temp_exclude_fields = [k for (k, v) in self.getDiscretizerFields().items()] + self.getExcludeFields()

        def columns(its):
            cols = []
            for sf in st.fields:
                for it in its:
                    if sf.name not in temp_exclude_fields and isinstance(sf.dataType, it):
                        cols.append(sf.name)
            return cols

        int_columns = columns([IntegerType, LongType, ShortType, DecimalType])
        array_columns = columns([ArrayType])
        vector_columns = columns([VectorUDT])
        float_columns = columns([FloatType, DoubleType])
        string_or_text_columns = columns([StringType])

        suffix = "_EasyFeature"

        if len(self.getDiscretizerFields()) > 0:
            tmp_fields = [k for (k, v) in self.getDiscretizerFields().items()]
            imputer = Imputer(inputCols=tmp_fields, outputCols=[(item + "_tmp") for item in tmp_fields])
            df = imputer.fit(dataset).transform(dataset)
            discretizerPipeline = Pipeline(
                stages=[QuantileDiscretizer(numBuckets=v, inputCol=k + "_tmp", outputCol=(k + suffix)) for (k, v) in
                        self.getDiscretizerFields().items()])
            discretizerModels = discretizerPipeline.fit(df)
            df = discretizerModels.transform(df)
            self.discretizerFeatures = [(item.getInputCol(), item.getSplits()) for item in discretizerModels.stages]
            df.drop(*[(item + "_tmp") for item in tmp_fields])
        else:
            df = dataset

        # float columns will do missing value checking
        if len(float_columns) > 0:
            imputer = Imputer(inputCols=float_columns, outputCols=[(item + suffix) for item in float_columns])
            df = imputer.fit(dataset).transform(df)

        # find text fields and category fields
        def check_categorical_or_text_string():
            maybe_text_fields = ["body", "title", "content", "sentence", "summary"]
            if len(self.getTextFields()) > 0:
                return self.getTextFields()
            return [item for item in string_or_text_columns if item.lower() in maybe_text_fields]

        text_fields = check_categorical_or_text_string()
        string_columns = [item for item in string_or_text_columns if item not in text_fields]

        # all string fields except text fields will be treated as category feature
        if len(string_columns) > 0:
            cbt = CategoricalBinaryTransformer(inputCols=string_columns,
                                               outputCols=[(item + suffix) for item in string_columns],
                                               embeddingSize=12)
            df = cbt.transform(df)

        if len(text_fields) > 0:
            # analysis text fields
            analysis_fields = [(item + "_text") for item in text_fields]
            tat = TextAnalysisTransformer(inputCols=text_fields, outputCols=analysis_fields,
                                          stopwords=self.getStopwords(),
                                          textAnalysisParams=self.getTextAnalysisParams())
            df = tat.transform(df)

            # text fields will be processed as tfidf / embedding vector
            if self.getWordMode() == "tfidf":
                ttfdft = TextTFDFTransformer(inputCols=analysis_fields,
                                             outputCols=[(item + suffix) for item in analysis_fields],
                                             numFeatures=self.getNumFeatures())
                df = ttfdft.transform(df)

            # word embedding analysised text fields
            if self.getWordMode() == "embedding":
                test = TextEmbeddingSequenceTransformer(inputCols=analysis_fields,
                                                        outputCols=[(item + suffix) for item in analysis_fields],
                                                        embeddingSize=self.getEmbeddingSize(),
                                                        sequenceLength=self.getSequenceLength()
                                                        )
                df = test.transform(df)
                self.word_embedding_with_index = test.getWordEmbedding()
                self.w2v_model = test.w2v_model
                self.wordEmbeddingSavePath = test.getWordEmbeddingSavePath()

        if len(int_columns) > 0:
            # find who is category column in int columns
            assembler = VectorAssembler(inputCols=int_columns,
                                        outputCol="easy_feature_int_vector")
            df = assembler.transform(df)
            indexer = VectorIndexer(inputCol="easy_feature_int_vector", outputCol="int_vector_EasyFeature",
                                    maxCategories=self.getMaxCategories())
            indexerModel = indexer.fit(df)
            df = df.drop("easy_feature_int_vector", "int_vector_EasyFeature")
            self.categoricalFeatures = indexerModel.categoryMaps
            tmp_int_category = [int_columns[k] for (k, v) in self.categoricalFeatures.items()]
            tmp_int_non_category = [k for k in int_columns if k not in tmp_int_category]

            # assemble non-category columns
            assembler = VectorAssembler(inputCols=tmp_int_non_category,
                                        outputCol="int_vector_EasyFeature")
            df = assembler.transform(df)

            # onehot category int columns
            onehotPipeline = Pipeline(
                stages=[OneHotEncoder(inputCol=item, outputCol=(item + suffix)) for item in tmp_int_category])
            df = onehotPipeline.fit(df).transform(df)

        # if outputCol is specified then assemble all columns together
        if self.getOutputCol() is not None:
            should_assemble_columns = [item.name for item in df.schema if
                                       item.name.endswith(suffix)] + vector_columns
            if self.getWordMode() == "embedding":
                should_assemble_columns = [item for item in should_assemble_columns if
                                           not item.endswith("_text" + suffix)]
            print("should_assemble_columns: {}".format(should_assemble_columns))

            tmp_output_col = self.getOutputCol() + "_original" if self.getOutputColPNorm() is not None else self.getOutputCol()

            schema_mapping = dict([(item.name, item.dataType) for item in df.schema.fields])
            replace_columns = []
            for item in should_assemble_columns:
                if schema_mapping[item] == ArrayType(IntegerType(), True):
                    def dence(col):
                        return Vectors.dense(col)

                    dence_udf = fn.udf(dence, VectorUDT())
                    df = df.withColumn("tmp_" + item, dence_udf(df[item]).alias(item))
                    replace_columns.append(item)

            for item in replace_columns:
                should_assemble_columns.remove(item)
                should_assemble_columns.append("tmp_" + item)

            assembler = VectorAssembler(inputCols=should_assemble_columns,
                                        outputCol=tmp_output_col)
            df = assembler.transform(df)
            if self.getOutputColPNorm() is not None:
                df = Normalizer(inputCol=tmp_output_col, outputCol=self.getOutputCol(),
                                p=self.getOutputColPNorm()).transform(df)

            return df
