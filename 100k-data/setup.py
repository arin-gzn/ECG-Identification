# setup.py

from setuptools import setup

setup(name='preprocess-ecg-dependencies',
      version='0.1',
      description='Dependencies',
      install_requires=[
          'apache-beam[gcp]==2.26.0',
          'google-api-core',
          'google-apitools',
          'google-cloud-core',
          'googleapis-common-protos',
          'google-cloud-storage',
          'protobuf',
          'future',
          'biosppy',
          'numpy',
          'scipy',
          'pandas',
          'fsspec',
          'gcsfs'
      ]
      )

