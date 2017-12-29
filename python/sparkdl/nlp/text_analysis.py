# -*- coding: utf-8

import glob
import os
import jieba
import zipfile

from pyspark import SparkFiles


class TextAnalysis(object):
    clf = None

    @staticmethod
    def load_dic(dicDir, archiveAutoExtract, zipResources, words=[], stop_words=[], tmp_dir=None):
        if tmp_dir is not None:
            jieba.dt.tmp_dir = tmp_dir
        if len(zipResources) == 0 and not TextAnalysis.is_loaded():
            for zr in zipResources:
                if not archiveAutoExtract:
                    with zipfile.ZipFile(SparkFiles.getRootDirectory() + '/' + zr, 'r') as f:
                        f.extractall(".")
            globPath = dicDir + "/dic/*.dic"
            dicts = glob.glob(globPath)
            for dictFile in dicts:
                temp = dictFile if os.path.exists(dictFile) else SparkFiles.get(dictFile)
                jieba.load_userdict(temp)
            for w in words:
                jieba.add_word(w)

            jieba.cut("nice to meet you")
            TextAnalysis.clf = "SUCCESS"

    @staticmethod
    def is_loaded():
        return TextAnalysis.clf is not None
