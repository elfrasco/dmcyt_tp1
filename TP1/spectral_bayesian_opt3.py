#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:52:21 2019

@author: frasco
"""

# Números y Datos
import numpy as np
import pandas as pd

# Archivos
import urllib.request
import glob

# Análisis de sonido
import spotipy 

# Machine learning
# importar los paquetes para clustering
from sklearn.preprocessing import StandardScaler

# To work with categorical types
from pandas.api.types import CategoricalDtype

# Clustering (scipy)
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster

# Clustering (sklearn)
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics.pairwise import euclidean_distances

from funciones import plot_silhouette
from funciones import plot_silhouettes_and_sses
from funciones import get_silhouette_avg
from funciones import get_sse
from funciones import vanDongen

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, TSNE

from umap import UMAP

from hyperopt import hp, tpe, fmin
from hyperopt.pyll.base import scope

metadata = pd.read_csv('../data/metadata.csv', index_col='id')
audio_features = pd.read_csv('../data/audio_features.csv', index_col='id')
audio_analysis = pd.read_csv('../data/audio_analysis.csv', index_col='id')

audio_features = audio_features[['acousticness','danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness','loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]

# Para que las comparaciones sean del mismo largo,
# se remueve de audio_feature y metadata el track que no se encuentra en audio_analysis
merged = audio_features.merge(audio_analysis, how = 'left', on = 'id')
id_to_remove = merged[merged.timbre_mean_0.isnull()].index[0]

audio_features = audio_features.drop(id_to_remove, axis = 0)

audio_tracks = pd.merge(audio_features, audio_analysis, how = 'inner', on = 'id')
metadata = metadata.drop(id_to_remove, axis = 0)

# Se ordenan los datasets para que los tracks estén en el mismo orden
audio_features = audio_features.sort_index()
audio_analysis = audio_analysis.sort_index()
audio_tracks = audio_tracks.sort_index()
metadata = metadata.sort_index()

datasets = {
    "audio_features": audio_features,
    "audio_analysis": audio_analysis,
    "audio_tracks": audio_tracks
}

scalers = {
    "minMax": MinMaxScaler(feature_range=(0,1)),
    "standard": StandardScaler()
}

def vanDongenSpectral(args):
    
    neighbors, min_d, components, metric, dataset, scaler = args
    k = 5
    
    print(dataset + ', ' + metric + ', ' + scaler + ', n_components=' + str(components) + ', n_neighbors=' + str(neighbors) + ', min_dist=' + str(min_d) + ', k=' + str(k))
    
    # Se estandariza usando el scaler correspondiente
    df = scalers[scaler].fit_transform(datasets[dataset])
                            
    # Se aplica UMAP
    um = UMAP(n_components = components, n_neighbors = neighbors, min_dist = min_d, metric = metric)
    embedding = um.fit_transform(df)
                            
    # Se aplica KMeans al embedding
    km = KMeans(n_clusters = k, random_state = 0).fit(embedding)
                            
    # Se calcula la matriz de confusion
    tmp = pd.DataFrame({'Generos': metadata.genre, 'data': km.labels_})
    ct = pd.crosstab(tmp['Generos'], tmp['data'])
                            
    return vanDongen(ct)

space = [scope.int(hp.quniform('neighbors', 2, 200, 1)),
         hp.choice('min_d', [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]),
         scope.int(hp.quniform('components', 2, 16, 1)),
         hp.choice('metric', ["euclidean", "manhattan", "minkowski", "canberra", "mahalanobis", "cosine", "correlation"]),
         hp.choice('dataset', ["audio_features", "audio_analysis", "audio_tracks"]),
         hp.choice('scaler', ["minMax", "standard"])]

# {'components': 15.0, 'dataset': 0, 'k': 2.0, 'metric': 5, 'min_d': 1, 'neighbors': 4.0, 'scaler': 1}
# Iteration: 416
best = fmin(fn = vanDongenSpectral,
            space = space, algo = tpe.suggest, 
            max_evals = 1000)

print(best)
