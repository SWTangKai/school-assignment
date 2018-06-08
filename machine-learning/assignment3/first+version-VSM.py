
# coding: utf-8


import pandas as pd
from scipy.spatial import distance
import numpy as np
from utils import *


origin_data_df = pd.read_csv('./input.txt', sep='\t', header=-1)
origin_data_df.columns = ['time', 'content']

stopwords = set()
with open('./Chinese-StopWords.txt') as f:
    for i in f.readlines():
        stopwords.add(i.strip('\n'))

origin_data_df['remove_stop'] = origin_data_df.content.apply(lambda s: ' '.join(
    [x for x in s.split(' ') if x not in stopwords]))

tf_idf_matrix = GetTfidfMatrix(origin_data_df.content)

from sklearn import cluster

kmeans_cls = cluster.KMeans()
origin_data_df['kmeans_label'] = kmeans_cls.fit_predict(tf_idf_matrix)
