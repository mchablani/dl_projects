#
# from fast.ai course materials
#

import math, keras, datetime, pandas as pd, numpy as np, keras.backend as K, threading, json, re, collections
import tarfile, tensorflow as tf, matplotlib.pyplot as plt, operator, random, pickle, glob, os, bcolz # , xgboost
import shutil, sklearn, functools, itertools, scipy
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, LSHForest
import IPython
from IPython.display import display, Audio
from numpy.random import normal
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import ToktokTokenizer, StanfordTokenizer
from functools import reduce
from itertools import chain
from scipy.misc import imsave

from tensorflow.python.framework import ops
#from tensorflow.contrib import rnn, legacy_seq2seq as seq2seq

from keras_tqdm import TQDMNotebookCallback
from keras import initializations
from keras.applications.resnet50 import ResNet50, decode_predictions, conv_block, identity_block
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))