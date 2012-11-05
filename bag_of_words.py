#!/usr/bin/env python

import sys
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

if __name__ == "__main__":
    hogs = np.load(sys.argv[1])
    clusters = np.load(sys.argv[2])
    # Compute a distance matrix, from it get HOG labels
    labels = np.array(map(np.argmax,
                          pairwise_distances(hogs, clusters)))
    # Compute a label histogram
    hist = np.bincount(labels,minlength=len(clusters))
    print np.shape(hist)
    # Save to file
    np.save(sys.argv[3],hist)
