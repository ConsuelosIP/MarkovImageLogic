#!/usr/bin/env python
from PIL import Image
import sys
import os
import numpy as np
from skimage.feature import hog

if __name__ == "__main__":
    im = Image.open(sys.argv[1])
    im.load()

	# split the image into individual bands
    rgb = map(np.array,im.split())

	# Save the HOGs
    orts=9
    cpb=3
    hogs = np.hstack(map(lambda i:
                            np.reshape(
                                hog( i,
                                     normalise=True,
                                     orientations=orts,
                                     cells_per_block=(cpb,cpb)),
                                (-1,orts*(cpb**2))),
                     rgb))
    #print(sum(hogs[0,0,0].ravel()))
    print(np.shape(hogs))
    np.save(sys.argv[2],hogs)

	#baseName = os.path.basename(os.path.splitext(sys.argv[1])[0])
	#r.save("Red-" + baseName + ".pgm")
	#b.save("Green-" + baseName + ".pgm")
	#g.save("Blue-" + baseName + ".pgm")
