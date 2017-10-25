import codecs
import os

from setuptools import setup
# See this web page for explanations:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
PACKAGES = ["sparkdl"]
KEYWORDS = ["spark", "deep learning", "distributed computing", "machine learning"]
CLASSIFIERS = [
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]
# Project root
ROOT = os.path.abspath(os.getcwd() + "/")


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(ROOT, *parts), "rb", "utf-8") as f:
        return f.read()

setup(
    name="spark-deep-learning",
    description="Integration tools for running deep learning on Spark",
    license="Apache 2.0",
    url="https://github.com/allwefantasy/spark-deep-learning",
    version="0.2.2",
    author="Joseph Bradley",
    author_email="joseph@databricks.com",
    maintainer="Tim Hunter",
    maintainer_email="timhunter@databricks.com",
    keywords=KEYWORDS,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
    zip_safe=False
)
