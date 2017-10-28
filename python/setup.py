import codecs
import os

from setuptools import setup, find_packages

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
ROOT = os.path.abspath(os.path.dirname(__file__))

#
#
# def read(*parts):
#     """
#     Build an absolute path from *parts* and and return the contents of the
#     resulting file.  Assume UTF-8 encoding.
#     """
#     with codecs.open(os.path.join(ROOT, *parts), "rb", "utf-8") as f:
#         return f.read()

#
# def configuration(parent_package='', top_path=None):
#     if os.path.exists('MANIFEST'):
#         os.remove('MANIFEST')
#
#     from numpy.distutils.misc_util import Configuration
#     config = Configuration(None, parent_package, top_path)
#
#     # Avoid non-useful msg:
#     # "Ignoring attempt to set 'name' (from ... "
#     config.set_options(ignore_setup_xxx_py=True,
#                        assume_default_configuration=True,
#                        delegate_options_to_subpackages=True,
#                        quiet=True)
#
#     config.add_subpackage('sparkdl')
#
#     return config


setup(
    name="sparkdl",
    description="Integration tools for running deep learning on Spark",
    license="Apache 2.0",
    url="https://github.com/allwefantasy/spark-deep-learning",
    version="0.2.2",
    author="Joseph Bradley",
    author_email="joseph@databricks.com",
    maintainer="Tim Hunter",
    maintainer_email="timhunter@databricks.com",
    keywords=KEYWORDS,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    zip_safe=False,
    include_package_data=True
)
