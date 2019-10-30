#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pylab

# Números y Datos
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import random

from sklearn.neighbors import NearestNeighbors

###########################################################################
# Van Dongen
def vanDongen(ct):
    n2=2*(sum(ct.apply(sum,axis=1)))
    sumi = sum(ct.apply(np.max,axis=1))
    sumj = sum(ct.apply(np.max,axis=0))
    maxsumi = np.max(ct.apply(sum,axis=1))
    maxsumj = np.max(ct.apply(sum,axis=0))
    vd = (n2 - sumi - sumj)/(n2 - maxsumi - maxsumj)
    return vd
    
###########################################################################
# Plot Silhouette
def plot_silhouette(df, k):

    # Se calcula KMeans
    km = KMeans(n_clusters = k, random_state = 10).fit(df)
    silhouette_avg = silhouette_score(df, km.labels_)
    
    # Se calcula el silhouette de cada observación
    sample_silhouette_values = silhouette_samples(df, km.labels_)
    
    # Se estima el coeficiente de Silhouette para cada cluster
    cluster_labels = km.labels_
    n_clusters = len(np.unique(cluster_labels))
    ith_cluster_silhouette_avg = []
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_avg.append(np.mean(ith_cluster_silhouette_values))
    
    # Se grafican los coeficientes de silhouette obtenidos
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Marcar los graficos de Silhouette con el numero de cluster
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Marcar los graficos de Silhouette con el Silhuette promedio del cluster
        ax1.text(0.6, y_lower + 0.5 * size_cluster_i, str(round(ith_cluster_silhouette_avg[i],3)))

        # Calcular donde comenzar el proximo grafixo
        y_lower = y_upper + 10 # Marco una distancia de 10 entre graficos para que haya un espacio

    ax1.set_xlabel("Coeficiente de silhouette")
    ax1.set_ylabel("Cluster label")

    # La linea vertical es el Silhouette promedio
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
    
    
###########################################################################
# Plot Silhouettes and SSEs
def plot_silhouettes_and_sses(df, kmax):
    kmeans = []
    sse = []
    sil = []
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters = k, random_state = 10).fit(df)
        kmeans.append(km)
        sil.append(silhouette_score(df, km.labels_))
        sse.append(km.inertia_)
    
    x = np.arange(2, kmax + 1)

    # Se abre una figura nueva
    fig = pylab.figure()
    
    # Se grafican los silhouttes
    axsil = fig.add_axes([0.05,0.05,0.9,0.475])
    axsil.set_ylabel("Silhoutte")
    axsil.set_xlabel("k")
    plt.plot(x, sil)
    
    # Se grafican los SSE
    axsse = fig.add_axes([0.05,0.525,0.9,0.475])
    axsse.set_ylabel("SSE")
    plt.plot(x, sse)
    return sse, sil


def list_plot(xlabel, ylabel, fig, ax, data):
    #ax_label = fig.add_axes([0.05,0.05,0.9,0.475])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)
    for x in range(len(data)):
        ax.step(np.arange(2, len(data[x][0]) + 2), data[x][0], label=data[x][1])
        ax.plot(np.arange(2, len(data[x][0]) + 2), data[x][0], 'C' + str(x) + 'o', alpha=0.5)  


###########################################################################
#Plot se grafican los sse y coeficientes de silhoutte de varios datasets
import matplotlib.gridspec as gridspec

def plot_all_silhouettes_and_sses(data_sil, data_sse):
    # Se abre una figura nueva
    fig = plt.figure()
    gs1 = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    # Se grafican los silhouttes
    list_plot('k', 'Silhoutte', fig, ax1, data_sil)
    list_plot('k', 'SSE', fig, ax2, data_sse)
    ax2.legend()
    gs1.tight_layout(fig)
    


###########################################################################
# Silhouette Average
def get_silhouette_avg(df, k):
    km = KMeans(n_clusters = k, random_state = 0).fit(df)
    return silhouette_score(df, km.labels_)

def get_sse(df, k):
    km = KMeans(n_clusters = k, random_state = 0).fit(df)
    return km.inertia_

###########################################################################
# Tendencia al clustering (Hopkins)
def hopkins(df,*args):
    n = df.shape[0] # filas
    d = df.shape[1] # columnas
    if not args:
        print("Numero de puntos al azar por defecto")
        m = int(0.1 * n) # cantidad de puntos al azar (default)
    else:
        m = args[0] # cantidad de puntos al azar

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(df) # buscador de vecinos

    rand_ind = random.sample(range(0, n, 1), m) # indices al azar

    ui = []
    wi = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.normal(size=(1, d)).reshape(1, -1), 2, return_distance=True) # distancia a los nuevos puntos
        ui.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(df[rand_ind[j]].reshape(1, -1), 2, return_distance=True) # distancia a los puntos al azar
        wi.append(w_dist[0][1])

    H = sum(wi) / (sum(ui) + sum(wi))
    return H
