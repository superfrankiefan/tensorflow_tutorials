import collections
import numpy as np
import pandas as pd
import tensorflow as tf

# Check TensorFlow Version
print('TensorFlow Version:', tf.__version__)

# download data using keras API
census_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
census_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
census_train_path =