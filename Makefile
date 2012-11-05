.PRECIOUS: *.npy
JPEGS=$(wildcard Stanford/*.jpg)
HOGS=$(patsubst Stanford/%.jpg, hogs/%-hog.npy, $(JPEGS))
PYTHON=. venv/bin/activate  && python
HOG=$(PYTHON) hog.py

all : 500-means.npy 1000-means.npy

Stanford40_JPEGImages.zip :
	curl http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip -o $@

JPEGImages : Stanford40_JPEGImages.zip 
	rm -rf $@
	unzip $<

Stanford : JPEGImages
	rm -rf $@
	ln -s $< $@
	
%-means.npy : $(HOGS)
	$(PYTHON) sample_k_means.py $(@:-means.npy=) 0.05 243 $@ 'hogs/*-hog.npy'

hogs/%-hog.npy : Stanford/%.jpg
	@ mkdir -p hogs
	$(PYTHON) hog.py $< $@

clean :
	rm -f *.png *.pgm *.hog *.npy
