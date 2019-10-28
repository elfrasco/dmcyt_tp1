#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:52:21 2019

@author: frasco
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['font.size'] = 22
from IPython.display import Audio, Markdown, Image
import pylab
import seaborn as sns

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

genres = []
for genre in metadata.genre:
    if genre == 'ambient':
        genres.append(0)
    if genre == 'classical':
        genres.append(1)
    if genre == 'drum-and-bass':
        genres.append(2)
    if genre == 'jazz':
        genres.append(3)
    if genre == 'world-music':
        genres.append(4)

columns = ['Dataset', 'Metric', 'Scaler',
           'n_components', 'min_dist', 'n_neighbors',
           'k', 'silhoutte', 'sse', 'vanDongen', 'adjRand']
results = []

n_neighbors = [2, 5, 10, 20, 50, 100, 200]

min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]

n_components = range(2, 16)

#metrics = ["euclidean", "manhattan", "minkowski", "canberra", "mahalanobis", "cosine", "correlation"]

metrics = ["canberra"]

datasets = {
    "audio_features": audio_features,
    "audio_analysis": audio_analysis,
    "audio_tracks": audio_tracks
}

scalers = {
    "minMax": MinMaxScaler(feature_range=(0,1)),
    "standard": StandardScaler()
}

ks = range(2, 16)

for dataset in datasets.keys():
    for metric in metrics:
        for scaler in scalers.keys():
            for components in n_components:
                for neighbors in n_neighbors:
                    for min_d in min_dist:
                        for k in ks:
                            
                            print(dataset + ', ' + metric + ', ' + scaler + ', n_components=' + str(components) + ', n_neighbors=' + str(neighbors) + ', min_dist=' + str(min_d) + ', k=' + str(k))

                            # Se estandariza usando el scaler correspondiente
                            df = scalers[scaler].fit_transform(datasets[dataset])
                            
                            # Se aplica UMAP
                            um = UMAP(n_components = components, n_neighbors = neighbors, min_dist = min_d, metric = metric)
                            embedding = um.fit_transform(df)
                            
                            # Se calculan las validaciones internas
                            sil = get_silhouette_avg(embedding, k)
                            sse = get_sse(embedding, k)
                            
                            # Se aplica KMeans
                            km = KMeans(n_clusters = k, random_state = 0).fit(embedding)
                            
                            # Se calcula la matriz de confusión
                            tmp = pd.DataFrame({'Generos': metadata.genre, 'data': km.labels_})
                            ct = pd.crosstab(tmp['Generos'], tmp['data'])
                            
                            # Se calculan las validaciones externas
                            vd = vanDongen(ct)
                            rand = adjusted_rand_score(metadata.genre, km.labels_)
                            
                            # Se guardan los resultados de la corrida
                            results.append([dataset, metric, scaler,
                                           components, min_d, neighbors,
                                           k, sil, sse, vd, rand])

df_results = pd.DataFrame(results, columns = columns)
df_results

df_results.to_csv('spectral_' + metrics[0] +'.csv', index = False)
