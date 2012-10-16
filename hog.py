#!/usr/bin/env python
from PIL import Image, ImageEnhance, ImageChops
import sys
import os
import numpy as np
from skimage.feature import hog

if __name__ == "__main__":
	im = Image.open(sys.argv[1])

	# split the image into individual bands
	rgb = map(np.array,im.split())

	# Save the HOGs
	hogs = np.array(map(lambda i: hog(i,normalise=True), rgb))
	np.save(sys.argv[2],hogs)

	#baseName = os.path.basename(os.path.splitext(sys.argv[1])[0])
	#r.save("Red-" + baseName + ".pgm")
	#b.save("Green-" + baseName + ".pgm")
	#g.save("Blue-" + baseName + ".pgm")
