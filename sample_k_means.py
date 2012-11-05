#!/usr/bin/env python

import sys
import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def load_fraction(fn, frac=.1, shape=None) :
    "Loads a random fraction of vectors from a .npy file"
    try : 
        if shape :
            data = np.reshape(np.load(fn),shape)
        else :
            data = np.load(fn)
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        return data[idx[:int(len(data)*frac)]]
    except :
        print "Could not load %s" % fn
        return np.zeros(0)

if __name__ == "__main__":
    n_clusters = int(sys.argv[1])
    frac = float(sys.argv[2])
    shape = (-1,int(sys.argv[3]))
    outfn = sys.argv[4]
    files = glob.glob(sys.argv[5])

    print "Loading %i files" % len(files)
    data = np.vstack(map(lambda fn : load_fraction(fn, frac=frac, shape=shape), 
                         files))
    print "Loaded %i vectors from %i files" % (len(data), len(files))

    # Use MiniBatch KMeans to extract centers
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
                          batch_size=45, n_init=10, max_no_improvement=10,
                          verbose=0)
    mbk.fit(data)
    print "Extracted %i clusters" % n_clusters

    # Save results
    np.save(outfn, mbk.cluster_centers_)
