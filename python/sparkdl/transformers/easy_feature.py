from pyspark.ml import Estimator, Transformer
from pyspark.ml.feature import Imputer, Param, Params, TypeConverters, VectorAssembler, VectorIndexer, Tokenizer, \
    HashingTF
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import *

from sparkdl.param.shared_params import HasEmbeddingSize, HasSequenceLength, HasOutputCol
from sparkdl.param import keyword_only
from sparkdl.transformers.tf_text import CategoricalBinaryTransformer, TextAnalysisTransformer, \
    TextEmbeddingSequenceTransformer, TextTFDFTransformer


class EasyFeature(Transformer, HasEmbeddingSize, HasSequenceLength, HasOutputCol):
    @keyword_only
    def __init__(self, textFields=None, wordMode=None, numFeatures=10000, maxCategories=20, embeddingSize=100,
                 sequenceLength=64,
                 wordEmbeddingSavePath=None, outputCol=None):
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
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, textFields=None, wordMode=None, numFeatures=10000, maxCategories=20, embeddingSize=100,
                  sequenceLength=64,
                  wordEmbeddingSavePath=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    textFields = Param(Params._dummy(), "textFields", "textFields",
                       typeConverter=TypeConverters.identity)

    wordMode = Param(Params._dummy(), "wordMode",
                     "wordMode: embedding or tfidf", typeConverter=TypeConverters.toString)
    textAnalysisParams = Param(Params._dummy(), "textAnalysisParams", "text analysis params",
                               typeConverter=TypeConverters.identity)
    wordEmbeddingSavePath = Param(Params._dummy(), "wordEmbeddingSavePath", "",
                                  typeConverter=TypeConverters.toString)
    numFeatures = Param(Params._dummy(), "numFeatures", "numFeatures",
                        typeConverter=TypeConverters.toInt)

    maxCategories = Param(Params._dummy(), "maxCategories",
                          "Threshold for the number of values a categorical feature can take " +
                          "(>= 2). If a feature is found to have > maxCategories values, then " +
                          "it is declared continuous.", typeConverter=TypeConverters.toInt)

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

    def getIgnoredColumns(self):
        return self.ignoredColumns

    def _transform(self, dataset):
        st = dataset.schema

        def columns(its):
            cols = []
            for sf in st.fields:
                for it in its:
                    if isinstance(sf.dataType, it):
                        cols.append(sf.name)
            return cols

        int_columns = columns([IntegerType, LongType, ShortType, DecimalType])
        array_columns = columns([ArrayType])
        vector_columns = columns([VectorUDT])
        float_columns = columns([FloatType, DoubleType])
        string_or_text_columns = columns([StringType])

        suffix = "_EasyFeature"

        # float columns will do missing value checking
        imputer = Imputer(inputCols=float_columns, outputCols=[(item + suffix) for item in float_columns])
        df = imputer.fit(dataset).transform(dataset)

        # find text fields and category fields
        def check_categorical_or_text_string():
            if len(self.getTextFields()) > 0:
                return self.getTextFields()
            return []

        text_fields = check_categorical_or_text_string()
        string_columns = [item for item in string_or_text_columns if item not in text_fields]

        # all string fields except text fields will be treated as category feature
        cbt = CategoricalBinaryTransformer(inputCols=string_columns,
                                           outputCols=[(item + suffix) for item in string_columns],
                                           embeddingSize=12)
        df = cbt.transform(df)

        # analysis text fields
        analysis_fields = [(item + "_text") for item in text_fields]
        tat = TextAnalysisTransformer(inputCols=text_fields, outputCols=analysis_fields)
        df = tat.transform(df)

        # text fields will be processed as tfidf / embedding vector
        if self.getWordMode() == "tfidf":
            ttfdft = TextTFDFTransformer(inputCols=analysis_fields,
                                         outputCols=[(item + suffix) for item in analysis_fields],
                                         numFeatures=10000)
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

        # int columns will be assembled simply and determine which are category feature with MaxCategories
        assembler = VectorAssembler(inputCols=int_columns,
                                    outputCol="easy_feature_int_vector")
        df = assembler.transform(df)
        indexer = VectorIndexer(inputCol="easy_feature_int_vector", outputCol="easy_feature_int_vector_2",
                                maxCategories=self.getMaxCategories())
        indexerModel = indexer.fit(df)
        self.categoricalFeatures = indexerModel.categoryMaps
        df = indexerModel.transform(df)

        # if outputCol is specified then assemble all columns together
        if self.getOutputCol() is not None:
            should_assemble_columns = [item.name for item in df.schema if
                                       item.name.endswith(suffix)] + vector_columns
            if self.getWordMode() == "embedding":
                should_assemble_columns = [item for item in should_assemble_columns if
                                           not item.endswith("_text" + suffix)]
            print("should_assemble_columns: {}".format(should_assemble_columns))
            assembler = VectorAssembler(inputCols=should_assemble_columns,
                                        outputCol=self.getOutputCol())
            df = assembler.transform(df)

        return df
