#!/usr/bin/env python
import sys
import numpy as np
from numpy.random import multivariate_normal as gauss

# Dump to PDF
import matplotlib
matplotlib.use('PDF')
import pylab as pl

# Get K-Means
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":
    # Make two random clusters
    clst1 = np.array([gauss([2,2],[[.5,0],[0,.5]]) for i in range(100)])
    clst2 = np.array([gauss([-2,-2],[[.5,0],[0,.5]]) for i in range(100)])
    n_clusters = 2

    # Merge them together and shuffle
    data = np.vstack([clst1,clst2])
    np.random.shuffle(data)

    # Use MiniBatch KMeans to Extract
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, 
                          batch_size=45, n_init=10, max_no_improvement=10, 
                          verbose=0)

    mbk.fit(data)

    # Plot results
    colors = ['#4EACC5', '#FF9C34']
    for k in range(n_clusters):
        members = mbk.labels_ == k
        cluster_center = mbk.cluster_centers_[k]
        pl.plot(data[members, 0], data[members, 1], 'w',
                markerfacecolor=colors[k], marker='.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', 
                markerfacecolor=colors[k], markeredgecolor='k', markersize=6)
    
    #pl.plot(clst1[:,0], clst1[:,1], 'ro')
    #pl.plot(clst2[:,0], clst2[:,1], 'bo')
    #pl.xlabel('x')
    #pl.ylabel('y')
    pl.xlim(-10.0,10.0)
    pl.ylim(-10.0,10.0)
    pl.savefig(sys.argv[1])
